import 'dart:async';

import 'package:deepguard/models/connection_settings.dart';
import 'package:deepguard/screens/connection_settings_screen.dart';
import 'package:deepguard/services/api_service.dart';
import 'package:deepguard/services/connection_settings_store.dart';
import 'package:deepguard/widgets/backend_status_card.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'support/connection_test_doubles.dart';

Future<void> tapText(WidgetTester tester, String text) async {
  if (find.text(text).evaluate().isEmpty) {
    // ListView builds distant children only when they enter the viewport.
    await tester.scrollUntilVisible(
      find.text(text),
      200,
      scrollable: find.byType(Scrollable).first,
    );
  }
  await tester.ensureVisible(find.text(text).last);
  await tester.tap(find.text(text).last);
  await tester.pumpAndSettle();
}

Widget settingsApp(ConnectionSettingsStore store, HealthChecker checker) =>
    MaterialApp(
      home: ConnectionSettingsScreen(store: store, healthChecker: checker),
    );

void main() {
  testWidgets(
    'test checks draft only; save applies address and restart restores it',
    (tester) async {
      final repo = MemorySettingsRepository();
      final store = ConnectionSettingsStore(repo);
      addTearDown(store.dispose);
      final checked = <ConnectionSettings>[];
      Future<BackendHealth> check(ConnectionSettings setting) async {
        checked.add(setting);
        return const BackendHealth(modelLoaded: true);
      }

      await tester.pumpWidget(settingsApp(store, check));
      await tester.pumpAndSettle();
      await tester.enterText(find.byKey(const Key('wifi_ip')), '192.168.1.90');
      await tapText(tester, 'Test connection');
      expect(checked.single.wifiIp, '192.168.1.90');
      expect(repo.writes, 0);
      expect(store.value.wifiIp, '192.168.100.29');
      expect(find.text('Service connected. Model ready.'), findsOneWidget);
      await tapText(tester, 'Save settings');
      expect(store.value.wifiIp, '192.168.1.90');
      await tester.pumpWidget(const SizedBox.shrink());
      final restarted = ConnectionSettingsStore(repo);
      addTearDown(restarted.dispose);
      await tester.pumpWidget(settingsApp(restarted, check));
      await tester.pumpAndSettle();
      expect(
        tester
            .widget<TextFormField>(find.byKey(const Key('wifi_ip')))
            .controller!
            .text,
        '192.168.1.90',
      );
      expect(checked, hasLength(1));
    },
  );
  testWidgets(
    'bad IP blocks requests; USB preserves last valid Wi-Fi address',
    (tester) async {
      final repo = MemorySettingsRepository();
      final store = ConnectionSettingsStore(repo);
      addTearDown(store.dispose);
      var requests = 0;
      await tester.pumpWidget(
        settingsApp(store, (_) async {
          requests++;
          return const BackendHealth(modelLoaded: true);
        }),
      );
      await tester.pumpAndSettle();
      await tester.enterText(
        find.byKey(const Key('wifi_ip')),
        'http://192.168.1.90:8000',
      );
      await tapText(tester, 'Save settings');
      await tapText(tester, 'Test connection');
      expect(repo.writes, 0);
      expect(requests, 0);
      expect(find.textContaining('Do not include'), findsOneWidget);
      await tapText(tester, 'USB');
      await tapText(tester, 'Save settings');
      expect(store.value.mode, ConnectionMode.usb);
      expect(store.value.wifiIp, '192.168.100.29');
    },
  );
  testWidgets('failed save retains previous endpoint and retry works', (
    tester,
  ) async {
    final repo = MemorySettingsRepository()..failSave = true;
    final store = ConnectionSettingsStore(repo);
    addTearDown(store.dispose);
    await tester.pumpWidget(
      settingsApp(store, (_) async => const BackendHealth(modelLoaded: true)),
    );
    await tester.pumpAndSettle();
    await tapText(tester, 'USB');
    await tapText(tester, 'Save settings');
    expect(store.value.mode, ConnectionMode.wifi);
    expect(
      find.textContaining('previous connection is still active'),
      findsOneWidget,
    );
    repo.failSave = false;
    await tapText(tester, 'Save settings');
    expect(store.value.mode, ConnectionMode.usb);
  });
  testWidgets('timeout is explained; retry distinguishes model not ready', (
    tester,
  ) async {
    final store = ConnectionSettingsStore(MemorySettingsRepository());
    addTearDown(store.dispose);
    var fail = true;
    await tester.pumpWidget(
      settingsApp(store, (_) async {
        if (fail) {
          throw TimeoutException('Offline');
        }
        return const BackendHealth(modelLoaded: false);
      }),
    );
    await tester.pumpAndSettle();
    await tapText(tester, 'Test connection');
    expect(find.textContaining('No reply within 8 seconds'), findsOneWidget);
    fail = false;
    await tapText(tester, 'Test connection');
    expect(
      find.text('Service connected. The model is not ready yet.'),
      findsOneWidget,
    );
  });
  testWidgets('editing draft clears previous health result', (tester) async {
    final store = ConnectionSettingsStore(MemorySettingsRepository());
    addTearDown(store.dispose);
    await tester.pumpWidget(
      settingsApp(store, (_) async => const BackendHealth(modelLoaded: true)),
    );
    await tester.pumpAndSettle();
    await tapText(tester, 'Test connection');
    expect(find.byKey(const Key('connection_test_result')), findsOneWidget);
    await tester.ensureVisible(find.byKey(const Key('wifi_ip')));
    await tester.enterText(find.byKey(const Key('wifi_ip')), '192.168.1.80');
    await tester.pump();
    expect(find.byKey(const Key('connection_test_result')), findsNothing);
  });
  testWidgets('late test response after leaving screen is ignored', (
    tester,
  ) async {
    final store = ConnectionSettingsStore(MemorySettingsRepository());
    addTearDown(store.dispose);
    final pending = Completer<BackendHealth>();
    await tester.pumpWidget(settingsApp(store, (_) => pending.future));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.text('Test connection'));
    await tester.tap(find.text('Test connection'));
    await tester.pump();
    await tester.pumpWidget(const MaterialApp(home: Text('Elsewhere')));
    pending.complete(const BackendHealth(modelLoaded: true));
    await tester.pumpAndSettle();
    expect(tester.takeException(), isNull);
  });
  testWidgets('old server response cannot overwrite new server readiness', (
    tester,
  ) async {
    final store = ConnectionSettingsStore(MemorySettingsRepository());
    addTearDown(store.dispose);
    final old = Completer<BackendHealth>();
    final next = Completer<BackendHealth>();
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: BackendStatusCard(
            store: store,
            healthChecker: (s) =>
                s.mode == ConnectionMode.wifi ? old.future : next.future,
          ),
        ),
      ),
    );
    await tester.pump();
    await store.save(
      ConnectionSettings(mode: ConnectionMode.usb, wifiIp: store.value.wifiIp),
    );
    await tester.pump();
    next.complete(const BackendHealth(modelLoaded: false));
    await tester.pumpAndSettle();
    old.complete(const BackendHealth(modelLoaded: true));
    await tester.pumpAndSettle();
    expect(
      find.text('The model is not ready. Check the backend setup.'),
      findsOneWidget,
    );
    expect(find.textContaining('Xception model is ready'), findsNothing);
    expect(find.text('USB • http://127.0.0.1:8000'), findsOneWidget);
  });
  testWidgets('small screen and enlarged text have no layout errors', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(360, 800);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    final store = ConnectionSettingsStore(MemorySettingsRepository());
    addTearDown(store.dispose);
    await tester.pumpWidget(
      MaterialApp(
        builder: (context, child) => MediaQuery(
          data: MediaQuery.of(context)
              .copyWith(textScaler: const TextScaler.linear(1.5)),
          child: child!,
        ),
        home: ConnectionSettingsScreen(
          store: store,
          healthChecker: (_) async => const BackendHealth(modelLoaded: true),
        ),
      ),
    );
    await tester.pumpAndSettle();
    await tapText(tester, 'Test connection');
    await tapText(tester, 'Save settings');
    expect(tester.takeException(), isNull);
  });
}
