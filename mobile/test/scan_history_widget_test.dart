import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:deepguard/models/analysis_result.dart';
import 'package:deepguard/models/scan_record.dart';
import 'package:deepguard/screens/scan_history_screen.dart';
import 'package:deepguard/services/scan_history_store.dart';
import 'package:deepguard/widgets/video_upload_card.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

class MemoryHistory extends ScanHistoryStore {
  final records = <String, ScanRecord>{};
  bool failSave = false;
  bool failRead = false;
  int saveCalls = 0;
  @override
  Future<List<ScanRecord>> list() async {
    if (failRead) {
      throw StateError('Read failure');
    }
    return records.values.toList()
      ..sort((a, b) => b.createdAtUtc.compareTo(a.createdAtUtc));
  }

  @override
  Future<void> save(ScanRecord record) async {
    saveCalls++;
    if (failSave) {
      throw StateError('Storage full');
    }
    records.putIfAbsent(record.id, () => record);
    notifyListeners();
  }

  @override
  Future<void> delete(String id) async {
    records.remove(id);
    notifyListeners();
  }

  @override
  Future<void> clear() async {
    records.clear();
    notifyListeners();
  }
}

// file_picker 13 uses an abstract base class, so the test double extends it.
final class TestFile extends PlatformFile {
  @override
  String get name => 'fixture.mp4';
  @override
  Uri get uri => Uri.file('/fixture.mp4');
  @override
  Never get xFile =>
      throw UnsupportedError('Not used by the injected analyser.');
  @override
  int? lengthSync() => 1;
  @override
  Future<int?> length() async => 1;
  @override
  Future<Uint8List> readAsBytes() async => Uint8List(1);
  @override
  Stream<Uint8List> readAsByteStream() => Stream.value(Uint8List(1));
}

AnalysisResult fixture([String name = 'single']) {
  final d = jsonDecode(
    File('test/fixtures/${name}_api.json').readAsStringSync(),
  ) as Map<String, dynamic>;
  return AnalysisResult.fromJson(d, expectedBytes: d['size_bytes'] as int);
}

Widget upload(MemoryHistory history, VideoAnalyser analyser) => MaterialApp(
  home: Scaffold(
    body: SingleChildScrollView(
      child: VideoUploadCard(
        file: TestFile(),
        historyStore: history,
        analyser: analyser,
      ),
    ),
  ),
);

void main() {
  testWidgets('completed response is saved once and reported as saved', (
    tester,
  ) async {
    final store = MemoryHistory();
    await tester.pumpWidget(
      upload(store, (_, {required abortTrigger}) async => fixture()),
    );
    await tester.tap(find.text('Analyse video').last);
    await tester.pumpAndSettle();
    expect(store.records, hasLength(1));
    expect(find.text('Saved to History on this device.'), findsOneWidget);
    expect(find.text('Prediction: Fake'), findsOneWidget);
  });

  testWidgets(
    'save failure preserves result and retry does not analyse again',
    (tester) async {
      final store = MemoryHistory()..failSave = true;
      var requests = 0;
      await tester.pumpWidget(
        upload(store, (_, {required abortTrigger}) async {
          requests++;
          return fixture();
        }),
      );
      await tester.tap(find.text('Analyse video').last);
      await tester.pumpAndSettle();
      expect(find.text('Prediction: Fake'), findsOneWidget);
      expect(find.text('Retry saving'), findsOneWidget);
      expect(store.records, isEmpty);
      store.failSave = false;
      await tester.ensureVisible(find.text('Retry saving'));
      await tester.tap(find.text('Retry saving'));
      await tester.pumpAndSettle();
      expect(requests, 1);
      expect(store.records, hasLength(1));
      expect(find.text('Saved to History on this device.'), findsOneWidget);
    },
  );

  testWidgets('network failure creates no history record', (tester) async {
    final store = MemoryHistory();
    await tester.pumpWidget(
      upload(store, (_, {required abortTrigger}) async {
        throw const SocketException('Disconnected');
      }),
    );
    await tester.tap(find.text('Analyse video').last);
    await tester.pumpAndSettle();
    expect(store.saveCalls, 0);
    expect(find.textContaining('Check the connection'), findsOneWidget);
  });

  testWidgets('late response after stop waiting is not saved', (tester) async {
    final store = MemoryHistory();
    final response = Completer<AnalysisResult>();
    await tester.pumpWidget(
      upload(store, (_, {required abortTrigger}) => response.future),
    );
    await tester.tap(find.text('Analyse video').last);
    await tester.pump();
    await tester.tap(find.text('Stop waiting'));
    response.complete(fixture());
    await tester.pumpAndSettle();
    expect(store.saveCalls, 0);
    expect(find.text('Prediction: Fake'), findsNothing);
  });

  testWidgets('insufficient evidence is saved without a verdict or score', (
    tester,
  ) async {
    final store = MemoryHistory();
    await tester.pumpWidget(
      upload(store, (_, {required abortTrigger}) async => fixture('blank')),
    );
    await tester.tap(find.text('Analyse video').last);
    await tester.pumpAndSettle();
    expect(store.records.values.single.result.fakeScore, isNull);
    expect(find.text('Insufficient evidence'), findsOneWidget);
  });

  testWidgets('history opens details and deletes only after confirmation', (
    tester,
  ) async {
    final store = MemoryHistory();
    await store.save(ScanRecord(createdAt: DateTime.now(), result: fixture()));
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(body: ScanHistoryScreen(store: store)),
      ),
    );
    await tester.pumpAndSettle();
    await tester.tap(find.text('single.mp4'));
    await tester.pumpAndSettle();
    expect(find.text('Saved scan'), findsOneWidget);
    await tester.tap(find.byTooltip('Delete saved result'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Cancel'));
    await tester.pumpAndSettle();
    expect(store.records, hasLength(1));
    await tester.tap(find.byTooltip('Delete saved result'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Delete'));
    await tester.pumpAndSettle();
    expect(find.text('No saved scans yet'), findsOneWidget);
    expect(store.records, isEmpty);
  });

  testWidgets(
    'clear all has a cancel option and removes history after confirmation',
    (tester) async {
      final store = MemoryHistory();
      await store.save(
        ScanRecord(createdAt: DateTime.now(), result: fixture()),
      );
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(body: ScanHistoryScreen(store: store)),
        ),
      );
      await tester.pumpAndSettle();
      await tester.tap(find.text('Clear all'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Cancel'));
      await tester.pumpAndSettle();
      expect(store.records, hasLength(1));
      await tester.tap(find.text('Clear all'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Delete all'));
      await tester.pumpAndSettle();
      expect(store.records, isEmpty);
      expect(find.text('No saved scans yet'), findsOneWidget);
    },
  );

  testWidgets('history read failure has a working retry', (tester) async {
    final store = MemoryHistory()..failRead = true;
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(body: ScanHistoryScreen(store: store)),
      ),
    );
    await tester.pumpAndSettle();
    expect(find.textContaining('could not be opened'), findsOneWidget);
    store.failRead = false;
    await tester.tap(find.text('Retry'));
    await tester.pumpAndSettle();
    expect(find.text('No saved scans yet'), findsOneWidget);
  });
}
