import 'dart:io';

import 'package:deepguard/models/connection_settings.dart';
import 'package:deepguard/models/scan_record.dart';
import 'package:deepguard/services/connection_settings_repository.dart';
import 'package:deepguard/services/connection_settings_store.dart';
import 'package:deepguard/services/scan_history_database.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as path;
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

import 'scan_history_database_test.dart' show sampleResult;
import 'support/connection_test_doubles.dart';

void main() {
  sqfliteFfiInit();
  test('normalizes private IPv4 and fixes USB endpoint', () {
    for (final ip in [
      '192.168.100.29',
      '10.0.0.8',
      '172.16.0.4',
      '172.31.1.1',
    ]) {
      final setting = ConnectionSettings(mode: ConnectionMode.wifi, wifiIp: ip);
      expect(setting.baseUrl, 'http://$ip:8000');
      expect(setting.endpoint('health').path, '/health');
    }
    expect(
      ConnectionSettings.normalizeWifiIp(' 192.168.001.029 '),
      '192.168.1.29',
    );
    final usb = ConnectionSettings(
      mode: ConnectionMode.usb,
      wifiIp: '192.168.1.20',
    );
    expect(usb.baseUrl, 'http://127.0.0.1:8000');
    expect(usb.endpoint('upload-video').path, '/upload-video');
  });
  test('rejects invalid addresses and malformed saved modes', () {
    for (final bad in [
      '',
      'http://192.168.1.2',
      '192.168.1.2:8000',
      '192.168.1.2/path',
      '192.168.1.999',
      '192.168.1',
      '0.0.0.0',
      '127.0.0.1',
      'example.com',
      '8.8.8.8',
      '172.32.0.1',
    ]) {
      expect(
        () => ConnectionSettings(mode: ConnectionMode.wifi, wifiIp: bad),
        throwsFormatException,
        reason: bad,
      );
    }
    expect(
      () => ConnectionSettings.fromDatabase({
        'mode': 'bad',
        'wifi_ip': '192.168.1.1',
      }),
      throwsFormatException,
    );
  });
  test('shares initial read and keeps current source endpoint', () async {
    final repo = MemorySettingsRepository();
    final store = ConnectionSettingsStore(repo);
    addTearDown(store.dispose);
    await Future.wait([store.ensureLoaded(), store.ensureLoaded()]);
    expect(repo.reads, 1);
    expect(store.value.baseUrl, 'http://192.168.100.29:8000');
  });
  test(
    'failed save retains current endpoint; successful retry survives restart',
    () async {
      final repo = MemorySettingsRepository();
      final store = ConnectionSettingsStore(repo);
      addTearDown(store.dispose);
      await store.ensureLoaded();
      final before = store.value;
      final usb = ConnectionSettings(
        mode: ConnectionMode.usb,
        wifiIp: before.wifiIp,
      );
      repo.failSave = true;
      await expectLater(store.save(usb), throwsStateError);
      expect(store.value, before);
      expect(store.isSaving, false);
      repo.failSave = false;
      await store.save(usb);
      final restarted = ConnectionSettingsStore(repo);
      addTearDown(restarted.dispose);
      await restarted.ensureLoaded();
      expect(restarted.value, usb);
    },
  );
  test(
    'failed read has a visible warning, cleared by successful save',
    () async {
      final store = ConnectionSettingsStore(
        MemorySettingsRepository()..failRead = true,
      );
      addTearDown(store.dispose);
      await store.ensureLoaded();
      expect(store.loadError, isNotNull);
      await store.save(ConnectionSettings.initial);
      expect(store.loadError, isNull);
    },
  );
  group('SQLite', () {
    late Directory temp;
    late ConnectionSettingsDatabase db;
    setUp(() async {
      temp = await Directory.systemTemp.createTemp('deepguard_connection_');
      db = ConnectionSettingsDatabase(
        factory: databaseFactoryFfi,
        databasePath: path.join(temp.path, 'deepguard_connection.db'),
      );
    });
    tearDown(() async {
      await db.close();
      await temp.delete(recursive: true);
    });
    test('USB and Wi-Fi persist across reopen and repeated switches', () async {
      expect(await db.load(), isNull);
      final wifi = ConnectionSettings(
        mode: ConnectionMode.wifi,
        wifiIp: '10.0.0.7',
      );
      await db.save(wifi);
      await db.close();
      expect(await db.load(), wifi);
      final usb = ConnectionSettings(
        mode: ConnectionMode.usb,
        wifiIp: wifi.wifiIp,
      );
      await db.save(usb);
      await db.close();
      expect(await db.load(), usb);
      await db.save(wifi);
      expect(await db.load(), wifi);
    });
    test(
      'settings changes preserve existing history and original decisions',
      () async {
        final history = ScanHistoryDatabase(
          factory: databaseFactoryFfi,
          databasePath: path.join(temp.path, 'deepguard_history.db'),
        );
        try {
          await history.save(
            ScanRecord(
              createdAt: DateTime.utc(2026, 11, 10),
              result: sampleResult('keep'),
            ),
          );
          await history.close();
          await db.save(
            ConnectionSettings(
              mode: ConnectionMode.usb,
              wifiIp: '192.168.1.20',
            ),
          );
          await db.close();
          expect((await history.list()).single.id, 'keep');
          expect((await history.list()).single.result.fakeScore, 0.72);
        } finally {
          await history.close();
        }
      },
    );
    test('failed database open can be retried', () async {
      await db.close();
      final blocker = File(path.join(temp.path, 'blocked'));
      await blocker.writeAsString('occupied');
      db = ConnectionSettingsDatabase(
        factory: databaseFactoryFfi,
        databasePath: path.join(blocker.path, 'settings.db'),
      );
      await expectLater(db.load(), throwsA(anything));
      await blocker.delete();
      await Directory(blocker.path).create();
      await db.save(ConnectionSettings.initial);
      expect(await db.load(), ConnectionSettings.initial);
    });
  });
}
