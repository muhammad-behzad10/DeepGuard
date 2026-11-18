import 'dart:async';

import 'package:flutter/material.dart';

import '../models/connection_settings.dart';
import '../services/api_service.dart';
import '../services/connection_settings_store.dart';

class ConnectionSettingsScreen extends StatefulWidget {
  final ConnectionSettingsStore? store;
  final HealthChecker? healthChecker;
  const ConnectionSettingsScreen({super.key, this.store, this.healthChecker});
  @override
  State<ConnectionSettingsScreen> createState() =>
      _ConnectionSettingsScreenState();
}

class _ConnectionSettingsScreenState extends State<ConnectionSettingsScreen> {
  final _formKey = GlobalKey<FormState>();
  final _ip = TextEditingController();
  late final ConnectionSettingsStore _store;
  ConnectionMode _mode = ConnectionSettings.initial.mode;
  bool _ready = false;
  bool _testing = false;
  bool _saving = false;
  BackendHealth? _health;
  ConnectionSettings? _tested;
  String? _testError;
  String? _saveError;
  String? _savedMessage;
  bool get _busy => _testing || _saving;

  @override
  void initState() {
    super.initState();
    _store = widget.store ?? connectionSettingsStore;
    unawaited(_load());
  }

  Future<void> _load() async {
    await _store.ensureLoaded();
    if (!mounted) {
      return;
    }
    _ip.text = _store.value.wifiIp;
    setState(() {
      _mode = _store.value.mode;
      _ready = true;
    });
  }

  void _edited() {
    _health = null;
    _tested = null;
    _testError = null;
    _saveError = null;
    _savedMessage = null;
  }

  ConnectionSettings? _candidate() {
    if (!_formKey.currentState!.validate()) {
      return null;
    }
    return ConnectionSettings(
      mode: _mode,
      wifiIp: _mode == ConnectionMode.wifi ? _ip.text : _store.value.wifiIp,
    );
  }

  Future<void> _test() async {
    if (_busy) {
      return;
    }
    final target = _candidate();
    if (target == null) {
      return;
    }
    setState(() {
      _testing = true;
      _health = null;
      _tested = null;
      _testError = null;
      _savedMessage = null;
    });
    BackendHealth? result;
    String? error;
    try {
      result =
          await (widget.healthChecker?.call(target) ??
              ApiService.checkHealth(settings: target));
    } on TimeoutException {
      error =
          'No reply within 8 seconds. Check the laptop backend and connection.';
    } on FormatException {
      error = 'This address did not return a valid DeepGuard health response.';
    } catch (_) {
      error = target.mode == ConnectionMode.usb
          ? 'Could not connect by USB. Check the cable, USB debugging, '
                'port forwarding and the running backend.'
          : 'Could not connect by Wi-Fi. Check the laptop IP, shared Wi-Fi '
                'network, backend and firewall.';
    }
    if (!mounted) {
      return;
    }
    setState(() {
      _testing = false;
      _health = result;
      _tested = target;
      _testError = error;
    });
  }

  Future<void> _save() async {
    if (_busy) {
      return;
    }
    final target = _candidate();
    if (target == null) {
      return;
    }
    setState(() {
      _saving = true;
      _saveError = null;
      _savedMessage = null;
    });
    try {
      await _store.save(target);
      if (mounted) {
        _ip.text = target.wifiIp;
        setState(() {
          _savedMessage = 'Connection settings saved on this device.';
        });
      }
    } catch (_) {
      if (mounted) {
        setState(() {
          _saveError =
              'Settings could not be saved. '
              'The previous connection is still active. Try again.';
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          _saving = false;
        });
      }
    }
  }

  @override
  void dispose() {
    _ip.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final health = _health;
    final errorStyle = TextStyle(color: Theme.of(context).colorScheme.error);
    return PopScope(
      canPop: !_saving,
      child: Scaffold(
        appBar: AppBar(title: const Text('Connection settings')),
        body: SafeArea(
          child: !_ready
              ? const Center(child: CircularProgressIndicator())
              : Form(
                  key: _formKey,
                  child: ListView(
                    padding: const EdgeInsets.all(20),
                    children: [
                      Text(
                        'Connect to your laptop',
                        style: Theme.of(context).textTheme.headlineSmall,
                      ),
                      const SizedBox(height: 8),
                      const Text(
                        'Choose how this phone reaches the analysis service. '
                        'The laptop backend must be running.',
                      ),
                      const SizedBox(height: 16),
                      Text(
                        'Active: ${_store.value.label} • ${_store.value.baseUrl}',
                        key: const Key('active_connection'),
                      ),
                      if (_store.loadError != null) ...[
                        const SizedBox(height: 12),
                        Text(_store.loadError!, style: errorStyle),
                      ],
                      const SizedBox(height: 20),
                      SegmentedButton<ConnectionMode>(
                        segments: const [
                          ButtonSegment(
                            value: ConnectionMode.usb,
                            icon: Icon(Icons.usb),
                            label: Text('USB'),
                          ),
                          ButtonSegment(
                            value: ConnectionMode.wifi,
                            icon: Icon(Icons.wifi),
                            label: Text('Wi-Fi'),
                          ),
                        ],
                        selected: {_mode},
                        onSelectionChanged: _busy
                            ? null
                            : (selected) {
                                setState(() {
                                  _mode = selected.single;
                                  _edited();
                                });
                              },
                      ),
                      const SizedBox(height: 20),
                      if (_mode == ConnectionMode.wifi) ...[
                        TextFormField(
                          key: const Key('wifi_ip'),
                          controller: _ip,
                          enabled: !_busy,
                          keyboardType: const TextInputType.numberWithOptions(
                            decimal: true,
                          ),
                          autocorrect: false,
                          decoration: const InputDecoration(
                            labelText: 'Laptop Wi-Fi IPv4 address',
                            hintText: '192.168.100.29',
                            helperText: 'IP address only. Port 8000 is added automatically.',
                            helperMaxLines: 2,
                            errorMaxLines: 4,
                            border: OutlineInputBorder(),
                          ),
                          validator: (value) {
                            try {
                              ConnectionSettings.normalizeWifiIp(value ?? '');
                              return null;
                            } on FormatException catch (error) {
                              return error.message;
                            }
                          },
                          onChanged: (_) => setState(_edited),
                        ),
                        const SizedBox(height: 12),
                        const Text(
                          'Connect both devices to the same Wi-Fi. Use the IPv4 address '
                          'under the laptop Wi-Fi adapter in ipconfig. The address can '
                          'change on another network.',
                        ),
                      ] else ...[
                        const SelectableText(
                          'USB address: http://127.0.0.1:8000',
                        ),
                        const SizedBox(height: 12),
                        const Text(
                          'Connect the USB cable and enable USB debugging. After '
                          'reconnecting the phone, run this on the laptop:',
                        ),
                        const SizedBox(height: 8),
                        const SelectableText('adb reverse tcp:8000 tcp:8000'),
                      ],
                      const SizedBox(height: 20),
                      if (_testing) ...[
                        const LinearProgressIndicator(),
                        const SizedBox(height: 8),
                        const Text('Testing connection...'),
                      ],
                      if (_testError != null) ...[
                        Text(_testError!, style: errorStyle),
                        const SizedBox(height: 12),
                      ],
                      if (health != null) ...[
                        Text(
                          !health.modelLoaded
                              ? 'Service connected. The model is not ready yet.'
                              : health.busy
                              ? 'Service connected. Another scan is in progress.'
                              : 'Service connected. Model ready.',
                          key: const Key('connection_test_result'),
                          style: const TextStyle(fontWeight: FontWeight.bold),
                        ),
                        Text('Checked: ${_tested!.baseUrl}'),
                        if (_tested != _store.value)
                          const Text(
                            'Save settings to use this connection for scans.',
                          ),
                        const SizedBox(height: 12),
                      ],
                      if (_saveError != null) ...[
                        Text(_saveError!, style: errorStyle),
                        const SizedBox(height: 12),
                      ],
                      if (_savedMessage != null) ...[
                        Text(_savedMessage!),
                        const SizedBox(height: 12),
                      ],
                      Wrap(
                        spacing: 12,
                        runSpacing: 8,
                        children: [
                          OutlinedButton.icon(
                            onPressed: _busy ? null : _test,
                            icon: const Icon(Icons.network_check),
                            label: const Text('Test connection'),
                          ),
                          FilledButton.icon(
                            onPressed: _busy ? null : _save,
                            icon: const Icon(Icons.save_outlined),
                            label: Text(
                              _saving ? 'Saving...' : 'Save settings',
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      const Text(
                        'Testing does not upload a video or save a new address. '
                        'Save settings to apply your choice. Saved history stays available '
                        'without a backend connection.',
                      ),
                    ],
                  ),
                ),
        ),
      ),
    );
  }
}
