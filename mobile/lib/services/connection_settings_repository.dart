import 'package:path/path.dart' as path;
import 'package:sqflite_common/sqlite_api.dart';

import '../models/connection_settings.dart';

abstract class ConnectionSettingsRepository {
  Future<ConnectionSettings?> load();
  Future<void> save(ConnectionSettings settings);
}

/// Separate database: no migration or modification of scan history.
class ConnectionSettingsDatabase implements ConnectionSettingsRepository {
  final DatabaseFactory factory;
  final String? databasePath;
  Future<Database>? _opening;
  ConnectionSettingsDatabase({required this.factory, this.databasePath});

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
        path.join(await factory.getDatabasesPath(), 'deepguard_connection.db');
    return factory.openDatabase(
      file,
      options: OpenDatabaseOptions(
        version: 1,
        onCreate: (db, version) async {
          await db.execute('''CREATE TABLE connection_settings (
          id INTEGER PRIMARY KEY CHECK (id = 1),
          mode TEXT NOT NULL CHECK (mode IN ('usb', 'wifi')),
          wifi_ip TEXT NOT NULL
        )''');
        },
      ),
    );
  }

  @override
  Future<ConnectionSettings?> load() async {
    final db = await _database();
    final rows = await db.query(
      'connection_settings',
      where: 'id = 1',
      limit: 1,
    );
    return rows.isEmpty ? null : ConnectionSettings.fromDatabase(rows.single);
  }

  @override
  Future<void> save(ConnectionSettings settings) async {
    final db = await _database();
    await db.insert(
      'connection_settings',
      settings.toDatabase(),
      conflictAlgorithm: ConflictAlgorithm.replace,
    );
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
