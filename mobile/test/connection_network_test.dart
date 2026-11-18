import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:deepguard/models/connection_settings.dart';
import 'package:deepguard/services/api_service.dart';
import 'package:deepguard/services/connection_settings_store.dart';
import 'package:deepguard/services/video_upload_service.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'support/connection_test_doubles.dart';

final class UploadTestFile extends PlatformFile {
  final int bytes;
  final Completer<void>? started;
  final Future<void>? release;
  UploadTestFile(this.bytes, {this.started, this.release});
  @override
  String get name => 'single.mp4';
  @override
  Uri get uri => Uri.file('/single.mp4');
  @override
  Never get xFile => throw UnsupportedError('Not used');
  @override
  int? lengthSync() => bytes;
  @override
  Future<int?> length() async {
    started?.complete();
    await release;
    return bytes;
  }

  @override
  Future<Uint8List> readAsBytes() async => Uint8List(bytes);
  @override
  Stream<Uint8List> readAsByteStream() => Stream.value(Uint8List(bytes));
}

void main() {
  final body = File('test/fixtures/single_api.json').readAsStringSync();
  final bytes = (jsonDecode(body) as Map<String, dynamic>)['size_bytes'] as int;
  test(
    'health and upload use the same persisted endpoint in both modes',
    () async {
      for (final mode in ConnectionMode.values) {
        final target = ConnectionSettings(mode: mode, wifiIp: '192.168.1.40');
        final store = ConnectionSettingsStore(
          MemorySettingsRepository()..saved = target,
        );
        final seen = <Uri>[];
        final client = MockClient((request) async {
          seen.add(request.url);
          if (request.url.path == '/health') {
            return http.Response(
              '{"status":"ok","service":"DeepGuard","model_loaded":true,"busy":false}',
              200,
            );
          }
          expect(request.method, 'POST');
          expect(
            request.headers['content-type'],
            contains('multipart/form-data'),
          );
          return http.Response(body, 200);
        });
        await ApiService.checkHealth(store: store, client: client);
        await VideoUploadService.upload(
          UploadTestFile(bytes),
          abortTrigger: Completer<void>().future,
          settingsStore: store,
          client: client,
        );
        expect(seen.map((u) => '${u.scheme}://${u.authority}'), [
          target.baseUrl,
          target.baseUrl,
        ]);
        store.dispose();
        client.close();
      }
    },
  );
  test('in-flight upload retains endpoint captured before asynchronous file access', () async {
    final store = ConnectionSettingsStore(MemorySettingsRepository());
    addTearDown(store.dispose);
    final started = Completer<void>();
    final release = Completer<void>();
    Uri? sent;
    final client = MockClient((request) async {
      sent = request.url;
      return http.Response(body, 200);
    });
    addTearDown(client.close);
    final upload = VideoUploadService.upload(
      UploadTestFile(bytes, started: started, release: release.future),
      abortTrigger: Completer<void>().future,
      settingsStore: store,
      client: client,
    );
    await started.future;
    await store.save(
      ConnectionSettings(mode: ConnectionMode.usb, wifiIp: '192.168.100.29'),
    );
    release.complete();
    await upload;
    expect(sent.toString(), 'http://192.168.100.29:8000/upload-video');
    expect(store.value.mode, ConnectionMode.usb);
  });
  test(
    'unreadable saved settings block video upload until a choice is saved',
    () async {
      final store = ConnectionSettingsStore(
        MemorySettingsRepository()..failRead = true,
      );
      addTearDown(store.dispose);
      var sent = false;
      final client = MockClient((_) async {
        sent = true;
        return http.Response('{}', 200);
      });
      addTearDown(client.close);
      await expectLater(
        VideoUploadService.upload(
          UploadTestFile(1),
          abortTrigger: Completer<void>().future,
          settingsStore: store,
          client: client,
        ),
        throwsA(isA<UploadException>()),
      );
      expect(sent, false);
    },
  );
  test('health rejects wrong service, malformed types, HTTP errors', () async {
    for (final body in [
      '{}',
      '{"status":"ok","service":"Other","model_loaded":true}',
      '{"status":"ok","service":"DeepGuard","model_loaded":"true"}',
      '{"status":"ok","service":"DeepGuard","model_loaded":true,"busy":"no"}',
    ]) {
      final client = MockClient((_) async => http.Response(body, 200));
      await expectLater(
        ApiService.checkHealth(
          settings: ConnectionSettings.initial,
          client: client,
        ),
        throwsFormatException,
      );
      client.close();
    }
    final client = MockClient((_) async => http.Response('{}', 503));
    await expectLater(
      ApiService.checkHealth(
        settings: ConnectionSettings.initial,
        client: client,
      ),
      throwsException,
    );
    client.close();
  });
  test('health timeout is bounded', () async {
    final never = Completer<http.Response>();
    final client = MockClient((_) => never.future);
    await expectLater(
      ApiService.checkHealth(
        settings: ConnectionSettings.initial,
        client: client,
        timeout: const Duration(milliseconds: 10),
      ),
      throwsA(isA<TimeoutException>()),
    );
    never.complete(http.Response('{}', 200));
    client.close();
  });
}
