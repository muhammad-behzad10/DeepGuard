import 'package:deepguard/models/connection_settings.dart';
import 'package:deepguard/services/connection_settings_repository.dart';

class MemorySettingsRepository implements ConnectionSettingsRepository {
  ConnectionSettings? saved;
  bool failRead = false;
  bool failSave = false;
  int reads = 0;
  int writes = 0;
  @override
  Future<ConnectionSettings?> load() async {
    reads++;
    if (failRead) {
      throw StateError('Storage unavailable');
    }
    return saved;
  }

  @override
  Future<void> save(ConnectionSettings settings) async {
    writes++;
    if (failSave) {
      throw StateError('Storage full');
    }
    saved = settings;
  }
}
