enum ConnectionMode { usb, wifi }

/// Connection configuration only; inference settings belong to the backend.
class ConnectionSettings {
  final ConnectionMode mode;
  final String wifiIp;
  const ConnectionSettings._(this.mode, this.wifiIp);

  // Preserve the working endpoint in the source uploaded on 10 Nov 2026.
  static const initial = ConnectionSettings._(
    ConnectionMode.wifi,
    '192.168.100.29',
  );

  factory ConnectionSettings({
    required ConnectionMode mode,
    required String wifiIp,
  }) => ConnectionSettings._(mode, normalizeWifiIp(wifiIp));

  static String normalizeWifiIp(String input) {
    final value = input.trim();
    if (!RegExp(r'^\d{1,3}(\.\d{1,3}){3}$').hasMatch(value)) {
      throw const FormatException(
        'Enter the laptop IPv4 address, for example '
        '192.168.100.29. Do not include http:// or :8000.',
      );
    }
    final parts = value.split('.').map(int.parse).toList();
    if (parts.any((part) => part > 255) ||
        !(parts[0] == 10 ||
            (parts[0] == 172 && parts[1] >= 16 && parts[1] <= 31) ||
            (parts[0] == 192 && parts[1] == 168))) {
      throw const FormatException(
        'Use the laptop private Wi-Fi IPv4 address '
        '(10.x.x.x, 172.16-31.x.x, or 192.168.x.x).',
      );
    }
    return parts.join('.');
  }

  String get baseUrl =>
      'http://${mode == ConnectionMode.usb ? '127.0.0.1' : wifiIp}:8000';
  String get label => mode == ConnectionMode.usb ? 'USB' : 'Wi-Fi';
  Uri endpoint(String name) {
    if (name != 'health' && name != 'upload-video') {
      throw ArgumentError.value(name, 'name', 'Unsupported endpoint');
    }
    return Uri.parse('$baseUrl/$name');
  }

  Map<String, Object?> toDatabase() => {
    'id': 1,
    'mode': mode.name,
    'wifi_ip': wifiIp,
  };
  factory ConnectionSettings.fromDatabase(Map<String, Object?> row) {
    final mode = row['mode'];
    final ip = row['wifi_ip'];
    if ((mode != 'usb' && mode != 'wifi') || ip is! String) {
      throw const FormatException('Saved connection settings are invalid.');
    }
    return ConnectionSettings(
      mode: mode == 'usb' ? ConnectionMode.usb : ConnectionMode.wifi,
      wifiIp: ip,
    );
  }
  @override
  bool operator ==(Object other) =>
      other is ConnectionSettings &&
      mode == other.mode &&
      wifiIp == other.wifiIp;
  @override
  int get hashCode => Object.hash(mode, wifiIp);
}
