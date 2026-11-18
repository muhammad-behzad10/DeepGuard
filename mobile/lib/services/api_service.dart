import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/connection_settings.dart';
import 'connection_settings_store.dart';

typedef HealthChecker = Future<BackendHealth> Function(
  ConnectionSettings settings,
);

class BackendHealth {
  final bool modelLoaded;
  final bool busy;
  const BackendHealth({required this.modelLoaded, this.busy = false});
}

class ApiService {
  static Future<BackendHealth> checkHealth({
    ConnectionSettings? settings,
    ConnectionSettingsStore? store,
    http.Client? client,
    Duration timeout = const Duration(seconds: 8),
  }) async {
    final ConnectionSettings target;
    if (settings != null) {
      // Testing a draft address does not need the saved-settings database.
      target = settings;
    } else {
      final activeStore = store ?? connectionSettingsStore;
      await activeStore.ensureLoaded();
      target = activeStore.value;
    }
    final transport = client ?? http.Client();
    try {
      final response = await transport
          .get(
            target.endpoint('health'),
            headers: {'Accept': 'application/json'},
          )
          .timeout(timeout);
      if (response.statusCode != 200) {
        throw Exception('Health check failed: HTTP ${response.statusCode}');
      }
      final Object? data = jsonDecode(utf8.decode(response.bodyBytes));
      if (data is! Map<String, dynamic> ||
          data['status'] != 'ok' ||
          data['service'] != 'DeepGuard' ||
          data['model_loaded'] is! bool ||
          (data.containsKey('busy') && data['busy'] is! bool)) {
        throw const FormatException('Unexpected backend response.');
      }
      return BackendHealth(
        modelLoaded: data['model_loaded'] as bool,
        busy: data['busy'] as bool? ?? false,
      );
    } finally {
      if (client == null) {
        transport.close();
      }
    }
  }
}
