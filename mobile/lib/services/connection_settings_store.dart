import 'package:flutter/foundation.dart';
import 'package:sqflite/sqflite.dart' show databaseFactory;

import '../models/connection_settings.dart';
import 'connection_settings_repository.dart';

class ConnectionSettingsStore extends ChangeNotifier {
  final ConnectionSettingsRepository repository;
  ConnectionSettings _value;
  Future<void>? _loading;
  bool _loaded = false;
  bool _saving = false;
  int _revision = 0;
  String? _loadError;
  ConnectionSettingsStore(
    this.repository, {
    ConnectionSettings initial = ConnectionSettings.initial,
  }) : _value = initial;

  ConnectionSettings get value => _value;
  bool get isLoaded => _loaded;
  bool get isSaving => _saving;
  int get revision => _revision;
  String? get loadError => _loadError;

  Future<void> ensureLoaded() {
    if (_loaded) {
      return Future<void>.value();
    }
    return _loading ??= _read();
  }

  Future<void> _read() async {
    try {
      final saved = await repository.load();
      if (saved != null) {
        _value = saved;
      }
      _loadError = null;
    } catch (_) {
      // Do not delete a failed database or stop offline history from opening.
      _loadError =
          'Saved connection settings could not be read. '
          'Review the address below and save your choice again.';
    } finally {
      _loaded = true;
      _loading = null;
      notifyListeners();
    }
  }

  Future<void> save(ConnectionSettings next) async {
    await ensureLoaded();
    if (_saving) {
      throw StateError('A connection setting is already being saved.');
    }
    _saving = true;
    notifyListeners();
    try {
      await repository.save(next);
      // The active endpoint changes only after persistence succeeds.
      _value = next;
      _loadError = null;
      _revision++;
    } finally {
      _saving = false;
      notifyListeners();
    }
  }
}

final connectionSettingsStore = ConnectionSettingsStore(
  ConnectionSettingsDatabase(factory: databaseFactory),
);
