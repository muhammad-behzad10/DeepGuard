import 'package:deepguard/main.dart';
import 'package:deepguard/services/api_service.dart';
import 'package:deepguard/services/connection_settings_store.dart';
import 'package:flutter_test/flutter_test.dart';

import 'scan_history_widget_test.dart' show MemoryHistory;
import 'support/connection_test_doubles.dart';

void main() {
  testWidgets('DeepGuard navigates to settings, history and about', (
    tester,
  ) async {
    final settings = ConnectionSettingsStore(MemorySettingsRepository());
    final history = MemoryHistory();
    addTearDown(settings.dispose);
    addTearDown(history.dispose);
    await tester.pumpWidget(
      MyApp(
        settingsStore: settings,
        historyStore: history,
        healthChecker: (_) async => const BackendHealth(modelLoaded: true),
      ),
    );
    await tester.pumpAndSettle();
    expect(find.text('Service connected'), findsOneWidget);
    await tester.tap(find.byTooltip('Connection settings'));
    await tester.pumpAndSettle();
    expect(find.text('Connect to your laptop'), findsOneWidget);
    await tester.tap(find.byTooltip('Back'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('History'));
    await tester.pumpAndSettle();
    expect(find.text('No saved scans yet'), findsOneWidget);
    await tester.tap(find.text('About'));
    await tester.pumpAndSettle();
    expect(find.text('About DeepGuard'), findsOneWidget);
    await tester.tap(find.text('Scan'));
    await tester.pumpAndSettle();
    expect(find.text('Service connected'), findsOneWidget);
  });
}
