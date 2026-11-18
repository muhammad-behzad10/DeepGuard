import 'dart:convert';

import 'package:file_picker/file_picker.dart';
import 'package:http/http.dart' as http;

import 'connection_settings_store.dart';
import '../models/analysis_result.dart';

class UploadException implements Exception {
  final String message;

  const UploadException(this.message);

  @override
  String toString() => message;
}

class VideoUploadService {
  static const int maxBytes = 30_000_000;

  static Future<AnalysisResult> upload(
    PlatformFile file, {
    required Future<void> abortTrigger,
    ConnectionSettingsStore? settingsStore,
    http.Client? client,
  }) async {
    if (file.extension?.toLowerCase() != 'mp4') {
      throw const UploadException('Please select an MP4 video.');
    }

    final store = settingsStore ?? connectionSettingsStore;
    await store.ensureLoaded();
    if (store.loadError != null) {
      throw const UploadException(
        'Open Connection settings and save the correct address before uploading.',
      );
    }
    // Snapshot before file access: an in-flight scan never changes destination.
    final endpoint = store.value.endpoint('upload-video');
    final bytes = await file.length().timeout(const Duration(seconds: 8));

    if (bytes == null || bytes <= 0) {
      throw const UploadException(
        'The file could not be read. Please select it again.',
      );
    }

    if (bytes > maxBytes) {
      throw const UploadException('The video exceeds the 30 MB limit.');
    }

    final transport = client ?? http.Client();

    try {
      final request = http.AbortableMultipartRequest(
        'POST',
        endpoint,
        abortTrigger: abortTrigger,
      );

      request.headers['Accept'] = 'application/json';

      request.files.add(
        http.MultipartFile(
          'file',
          file.readAsByteStream(),
          bytes,
          filename: file.name,
        ),
      );

      final response = await transport
          .send(request)
          .then(http.Response.fromStream)
          .timeout(const Duration(minutes: 3));

      if (response.statusCode != 200) {
        var message = 'Upload failed (HTTP ${response.statusCode}).';

        try {
          final Object? errorBody = jsonDecode(utf8.decode(response.bodyBytes));

          if (errorBody is Map<String, dynamic> &&
              errorBody['detail'] is String) {
            message = errorBody['detail'] as String;
          }
        } on FormatException {
          // Keep the HTTP error message if the body is not JSON.
        }

        throw UploadException(message);
      }

      final Object? data = jsonDecode(utf8.decode(response.bodyBytes));

      return AnalysisResult.fromJson(data, expectedBytes: bytes);
    } finally {
      if (client == null) {
        transport.close();
      }
    }
  }
}
