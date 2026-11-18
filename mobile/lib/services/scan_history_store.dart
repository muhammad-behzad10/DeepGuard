import 'package:flutter/foundation.dart';
import 'package:sqflite/sqflite.dart' show databaseFactory;

import '../models/scan_record.dart';
import 'scan_history_database.dart';

abstract class ScanHistoryStore extends ChangeNotifier {
  Future<List<ScanRecord>> list();
  Future<void> save(ScanRecord record);
  Future<void> delete(String id);
  Future<void> clear();
}

class LocalScanHistoryStore extends ScanHistoryStore {
  final ScanHistoryDatabase database;
  LocalScanHistoryStore(this.database);

  @override
  Future<List<ScanRecord>> list() => database.list();

  @override
  Future<void> save(ScanRecord record) async {
    await database.save(record);
    notifyListeners();
  }

  @override
  Future<void> delete(String id) async {
    await database.delete(id);
    notifyListeners();
  }

  @override
  Future<void> clear() async {
    await database.clear();
    notifyListeners();
  }
}

// Lazy initialization. A database failure must never prevent app startup.
final ScanHistoryStore scanHistoryStore = LocalScanHistoryStore(
  ScanHistoryDatabase(factory: databaseFactory),
);
