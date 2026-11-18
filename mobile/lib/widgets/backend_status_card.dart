import 'dart:async';

import 'package:flutter/material.dart';

import '../models/connection_settings.dart';
import '../screens/connection_settings_screen.dart';
import '../services/api_service.dart';
import '../services/connection_settings_store.dart';

class BackendStatusCard extends StatefulWidget {
  final ConnectionSettingsStore? store;
  final HealthChecker? healthChecker;
  const BackendStatusCard({super.key, this.store, this.healthChecker});
  @override
  State<BackendStatusCard> createState() => _BackendStatusCardState();
}

class _BackendStatusCardState extends State<BackendStatusCard> {
  late final ConnectionSettingsStore _store;
  ConnectionSettings? _target;
  int _revision = -1;
  bool _loading = true;
  BackendHealth? _health;
  String? _error;
  int _generation = 0;
  @override
  void initState() {
    super.initState();
    _store = widget.store ?? connectionSettingsStore;
    _store.addListener(_settingsChanged);
    unawaited(_start());
  }

  Future<void> _start() async {
    await _store.ensureLoaded();
    if (mounted && _target == null) {
      unawaited(_checkConnection());
    }
  }

  void _settingsChanged() {
    if (!mounted || !_store.isLoaded) {
      return;
    }
    if (_target != _store.value || _revision != _store.revision) {
      unawaited(_checkConnection());
    } else {
      setState(() {});
    }
  }

  Future<void> _checkConnection() async {
    if (!mounted || !_store.isLoaded) {
      return;
    }
    final target = _store.value;
    final generation = ++_generation;
    setState(() {
      _target = target;
      _revision = _store.revision;
      _loading = true;
      _health = null;
      _error = null;
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
      error = 'This address returned an unexpected service response.';
    } catch (_) {
      error = 'Could not reach the service. Check Connection settings and the laptop backend.';
    }
    if (!mounted || generation != _generation) {
      return;
    }
    setState(() {
      _health = result;
      _error = error;
      _loading = false;
    });
  }

  Future<void> _openSettings() async {
    await Navigator.of(context).push<void>(
      MaterialPageRoute(
        builder: (_) => ConnectionSettingsScreen(
          store: _store,
          healthChecker: widget.healthChecker,
        ),
      ),
    );
    if (mounted) {
      unawaited(_checkConnection());
    }
  }

  @override
  void dispose() {
    _generation++;
    _store.removeListener(_settingsChanged);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final connected = _health != null;
    final title = _loading
        ? 'Checking connection...'
        : connected
        ? 'Service connected'
        : 'Service unavailable';
    final description = _loading
        ? 'Contacting the analysis service.'
        : _error ??
              (_health?.modelLoaded != true
                  ? 'The model is not ready. Check the backend setup.'
                  : _health!.busy
                  ? 'Another scan is in progress. Please wait before analysing.'
                  : 'Xception model is ready for video analysis.');
    final color = _loading || connected
        ? Theme.of(context).colorScheme.primary
        : Theme.of(context).colorScheme.error;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  connected ? Icons.cloud_done_outlined : Icons.cloud_outlined,
                  color: color,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    title,
                    style: TextStyle(
                      color: color,
                      fontSize: 17,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Text(description),
            const SizedBox(height: 8),
            Text('${_store.value.label} • ${_store.value.baseUrl}'),
            if (_store.loadError != null) ...[
              const SizedBox(height: 8),
              Text(_store.loadError!),
            ],
            const SizedBox(height: 12),
            if (_loading) const LinearProgressIndicator(),
            Wrap(
              spacing: 8,
              runSpacing: 4,
              children: [
                TextButton.icon(
                  onPressed: _loading ? null : _checkConnection,
                  icon: const Icon(Icons.refresh),
                  label: Text(connected ? 'Check again' : 'Retry'),
                ),
                TextButton.icon(
                  onPressed: _openSettings,
                  icon: const Icon(Icons.settings_ethernet),
                  label: const Text('Connection settings'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
