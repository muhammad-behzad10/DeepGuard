import 'dart:io';

import 'package:deepguard/models/analysis_result.dart';
import 'package:deepguard/models/scan_record.dart';
import 'package:deepguard/services/scan_history_database.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as path;
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

AnalysisResult sampleResult(String id, {bool insufficient = false}) {
  return AnalysisResult(
    uploadId: id,
    filename: 'test_$id.mp4',
    sha256: 'a' * 64,
    sizeBytes: 100,
    status: insufficient ? 'insufficient_evidence' : 'analysed',
    label: insufficient ? null : 'Fake',
    reason: insufficient ? 'Only 3 usable faces.' : null,
    fakeScore: insufficient ? null : 0.72,
    sampledFrames: 16,
    usableFaces: insufficient ? 3 : 16,
    minimumFaces: 8,
    frameCounts: insufficient
        ? {'accepted': 3, 'multiple_faces': 13}
        : {'accepted': 16},
    processingSeconds: 32.5,
    apiVersion: '0.6.0',
    modelId: 'deepfakebench-xception-v1.0.1',
    checkpointSha256: 'b' * 64,
    preprocessingVersion: 'deepguard-yunet-xception-align-v1',
    decisionThreshold: insufficient ? null : 0.5,
  );
}

void main() {
  sqfliteFfiInit();
  late Directory temp;
  late ScanHistoryDatabase db;
  late String file;

  setUp(() async {
    temp = await Directory.systemTemp.createTemp('deepguard_history_test_');
    file = path.join(temp.path, 'history.db');
    db = ScanHistoryDatabase(factory: databaseFactoryFfi, databasePath: file);
  });
  tearDown(() async {
    await db.close();
    await temp.delete(recursive: true);
  });

  test(
    'result survives close and reopen with original score and provenance',
    () async {
      final created = DateTime.parse('2026-11-08T16:00:00+05:00');
      await db.save(
        ScanRecord(createdAt: created, result: sampleResult('persist')),
      );
      await db.close();
      db = ScanHistoryDatabase(factory: databaseFactoryFfi, databasePath: file);
      final saved = (await db.list()).single;
      expect(saved.createdAtUtc, DateTime.utc(2026, 11, 8, 11));
      expect(saved.result.fakeScore, 0.72);
      expect(saved.result.decisionThreshold, 0.5);
      expect(saved.result.modelId, 'deepfakebench-xception-v1.0.1');
      expect(saved.result.checkpointSha256, 'b' * 64);
      expect(saved.result.sha256, 'a' * 64);
      expect(saved.result.frameCounts, {'accepted': 16});
    },
  );

  test(
    'repeated save of one upload does not duplicate or change its date',
    () async {
      final first = ScanRecord(
        createdAt: DateTime.utc(2026, 11, 8),
        result: sampleResult('same'),
      );
      await Future.wait([db.save(first), db.save(first)]);
      await db.save(
        ScanRecord(
          createdAt: DateTime.utc(2026, 11, 9),
          result: sampleResult('same'),
        ),
      );
      final records = await db.list();
      expect(records, hasLength(1));
      expect(records.single.createdAtUtc, first.createdAtUtc);
      await db.save(
        ScanRecord(
          createdAt: DateTime.utc(2026, 11, 9),
          result: sampleResult('rescan'),
        ),
      );
      expect(
        await db.list(),
        hasLength(2),
      ); // Same input hash, genuinely new request.
    },
  );

  test('history is newest first after reopening', () async {
    for (final day in [7, 9, 8]) {
      await db.save(
        ScanRecord(
          createdAt: DateTime.utc(2026, 11, day),
          result: sampleResult('$day'),
        ),
      );
    }
    await db.close();
    expect((await db.list()).map((r) => r.id), ['9', '8', '7']);
  });

  test('insufficient evidence remains a result without a fake score', () async {
    await db.save(
      ScanRecord(
        createdAt: DateTime.now(),
        result: sampleResult('unclear', insufficient: true),
      ),
    );
    final result = (await db.list()).single.result;
    expect(result.hasPrediction, isFalse);
    expect(result.label, isNull);
    expect(result.fakeScore, isNull);
    expect(result.decisionThreshold, isNull);
    expect(result.reason, 'Only 3 usable faces.');
  });

  test(
    'single delete is parameterized and clear persists across reopen',
    () async {
      final ids = ['keep', "x' OR 1=1 --"];
      for (final id in ids) {
        await db.save(
          ScanRecord(createdAt: DateTime.now(), result: sampleResult(id)),
        );
      }
      await db.delete(ids.last);
      expect((await db.list()).single.id, 'keep');
      await db.clear();
      await db.close();
      expect(await db.list(), isEmpty);
      await db.save(
        ScanRecord(
          createdAt: DateTime.now(),
          result: sampleResult('after_clear'),
        ),
      );
      expect((await db.list()).single.id, 'after_clear');
    },
  );

  test(
    'stored records preserve an older model decision instead of recalculating',
    () async {
      final map = ScanRecord(
        createdAt: DateTime.now(),
        result: sampleResult('versioned'),
      ).toDatabase();
      map['model_id'] = 'a-previous-model';
      map['decision_threshold'] = 0.8;
      map['label'] = 'Real';
      await db.save(ScanRecord.fromDatabase(map));
      final result = (await db.list()).single.result;
      expect(result.label, 'Real');
      expect(result.fakeScore, 0.72);
      expect(result.decisionThreshold, 0.8);
      expect(result.modelId, 'a-previous-model');
    },
  );

  test(
    'invalid outcomes cannot enter history and existing rows remain intact',
    () async {
      await db.save(
        ScanRecord(createdAt: DateTime.now(), result: sampleResult('valid')),
      );
      final map = ScanRecord(
        createdAt: DateTime.now(),
        result: sampleResult('invalid'),
      ).toDatabase();
      map['fake_score'] = 1.5;
      await expectLater(
        db.save(ScanRecord.fromDatabase(map)),
        throwsA(isA<DatabaseException>()),
      );
      map['fake_score'] = null;
      await expectLater(
        db.save(ScanRecord.fromDatabase(map)),
        throwsA(isA<DatabaseException>()),
      );
      map['status'] = 'network_error';
      await expectLater(
        db.save(ScanRecord.fromDatabase(map)),
        throwsA(isA<DatabaseException>()),
      );
      expect((await db.list()).single.id, 'valid');
    },
  );

  test(
    'failed database open is retryable without automatic data deletion',
    () async {
      await db.close();
      final blocked = File(path.join(temp.path, 'blocked'));
      await blocked.writeAsString('not a directory');
      db = ScanHistoryDatabase(
        factory: databaseFactoryFfi,
        databasePath: path.join(blocked.path, 'history.db'),
      );
      await expectLater(db.list(), throwsA(anything));
      await blocked.delete();
      await Directory(blocked.path).create();
      await db.save(
        ScanRecord(
          createdAt: DateTime.now(),
          result: sampleResult('recovered'),
        ),
      );
      expect((await db.list()).single.id, 'recovered');
    },
  );
}
