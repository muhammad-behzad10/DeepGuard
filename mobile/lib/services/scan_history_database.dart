import 'package:path/path.dart' as path;
import 'package:sqflite_common/sqlite_api.dart';

import '../models/scan_record.dart';

/// SQLite implementation shared by the Android app and real-database tests.
/// Version 1 is new: future changes must add migrations, never drop history.
class ScanHistoryDatabase {
  final DatabaseFactory factory;
  final String? databasePath;
  Future<Database>? _opening;

  ScanHistoryDatabase({required this.factory, this.databasePath});

  Future<Database> _database() async {
    final opening = _opening ??= _open();
    try {
      return await opening;
    } catch (_) {
      if (identical(_opening, opening)) {
        _opening = null;
      }
      rethrow;
    }
  }

  Future<Database> _open() async {
    final file =
        databasePath ??
        path.join(await factory.getDatabasesPath(), 'deepguard_history.db');
    return await factory.openDatabase(
      file,
      options: OpenDatabaseOptions(
        version: 1,
        onCreate: (db, version) async {
          await db.execute('''
              CREATE TABLE scans (
                scan_id TEXT PRIMARY KEY NOT NULL,
                created_at_utc TEXT NOT NULL,
                filename TEXT NOT NULL,
                input_sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
                status TEXT NOT NULL CHECK (
                  status IN ('analysed', 'insufficient_evidence')),
                label TEXT,
                fake_score REAL,
                decision_threshold REAL,
                reason TEXT,
                api_version TEXT NOT NULL,
                model_id TEXT NOT NULL,
                checkpoint_sha256 TEXT NOT NULL,
                preprocessing_version TEXT NOT NULL,
                sampled_frames INTEGER NOT NULL CHECK (sampled_frames > 0),
                usable_faces INTEGER NOT NULL CHECK (
                  usable_faces >= 0 AND usable_faces <= sampled_frames),
                minimum_faces INTEGER NOT NULL CHECK (minimum_faces > 0),
                processing_seconds REAL NOT NULL CHECK (processing_seconds >= 0),
                frame_counts_json TEXT NOT NULL,
                schema_version INTEGER NOT NULL DEFAULT 1,
                CHECK (
                  (status = 'analysed' AND label IS NOT NULL AND
                   label IN ('Real', 'Fake') AND fake_score IS NOT NULL AND
                   fake_score >= 0 AND fake_score <= 1 AND
                   decision_threshold IS NOT NULL AND
                   decision_threshold >= 0 AND decision_threshold <= 1 AND
                   usable_faces >= minimum_faces AND
                   ((fake_score >= decision_threshold AND label = 'Fake') OR
                    (fake_score < decision_threshold AND label = 'Real')))
                  OR
                  (status = 'insufficient_evidence' AND label IS NULL AND
                   fake_score IS NULL AND decision_threshold IS NULL AND
                   usable_faces < minimum_faces AND reason IS NOT NULL AND
                   length(reason) > 0)
                )
              )
            ''');
          await db.execute(
            'CREATE INDEX scans_created ON scans(created_at_utc DESC)',
          );
        },
      ),
    );
  }

  Future<void> save(ScanRecord record) async {
    final db = await _database();
    // A retry uses the server upload ID, so it cannot create a duplicate scan.
    // IGNORE would also hide CHECK failures; use an explicit transaction.
    await db.transaction((txn) async {
      final existing = await txn.query(
        'scans',
        columns: ['scan_id'],
        where: 'scan_id = ?',
        whereArgs: [record.id],
        limit: 1,
      );
      if (existing.isEmpty) {
        await txn.insert('scans', record.toDatabase());
      }
    });
  }

  Future<List<ScanRecord>> list() async {
    final db = await _database();
    final rows = await db.query(
      'scans',
      orderBy: 'created_at_utc DESC, scan_id DESC',
    );
    return rows.map(ScanRecord.fromDatabase).toList(growable: false);
  }

  Future<void> delete(String scanId) async {
    final db = await _database();
    await db.delete('scans', where: 'scan_id = ?', whereArgs: [scanId]);
  }

  Future<void> clear() async {
    final db = await _database();
    await db.delete('scans');
  }

  Future<void> close() async {
    final opening = _opening;
    _opening = null;
    if (opening != null) {
      final db = await opening;
      await db.close();
    }
  }
}
