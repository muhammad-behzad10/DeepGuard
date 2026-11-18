import 'package:flutter/material.dart';

import '../models/scan_record.dart';
import '../services/scan_history_store.dart';
import 'scan_history_detail_screen.dart';

class ScanHistoryScreen extends StatefulWidget {
  final ScanHistoryStore? store;
  const ScanHistoryScreen({super.key, this.store});

  @override
  State<ScanHistoryScreen> createState() => _ScanHistoryScreenState();
}

class _ScanHistoryScreenState extends State<ScanHistoryScreen> {
  late final ScanHistoryStore _store;
  late Future<List<ScanRecord>> _records;
  bool _clearing = false;

  @override
  void initState() {
    super.initState();
    _store = widget.store ?? scanHistoryStore;
    _records = _store.list();
    _store.addListener(_reload);
  }

  void _reload() {
    if (!mounted) {
      return;
    }
    final records = _store.list();
    setState(() {
      _records = records;
    });
  }

  @override
  void dispose() {
    _store.removeListener(_reload);
    super.dispose();
  }

  Future<void> _clear() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear scan history?'),
        content: const Text(
          'Delete all saved scan results on this phone? This cannot be undone. '
          'Your original videos will not be deleted.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Delete all'),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) {
      return;
    }
    setState(() => _clearing = true);
    try {
      await _store.clear();
    } catch (_) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('History could not be cleared. Try again.'),
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() => _clearing = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<List<ScanRecord>>(
      future: _records,
      builder: (context, snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return const Center(child: CircularProgressIndicator());
        }
        if (snapshot.hasError) {
          return ListView(
            padding: const EdgeInsets.all(24),
            children: [
              const Icon(Icons.history, size: 56),
              const SizedBox(height: 16),
              const Text(
                'Saved history could not be opened. Your scans have not been '
                'deleted. You can still use the Scan tab.',
              ),
              const SizedBox(height: 12),
              OutlinedButton(onPressed: _reload, child: const Text('Retry')),
            ],
          );
        }
        final records = snapshot.data ?? [];
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 8),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Scan history',
                    style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Saved results on this device. History contains result '
                    'details, not copies of your videos.',
                  ),
                  const SizedBox(height: 8),
                  Row(
                    children: [
                      Expanded(child: Text('${records.length} saved scans')),
                      TextButton.icon(
                        onPressed: records.isEmpty || _clearing ? null : _clear,
                        icon: const Icon(Icons.delete_outline),
                        label: const Text('Clear all'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            if (_clearing) const LinearProgressIndicator(),
            Expanded(
              child: records.isEmpty
                  ? ListView(
                      padding: const EdgeInsets.all(28),
                      children: const [
                        Icon(Icons.history, size: 64),
                        SizedBox(height: 20),
                        Text(
                          'No saved scans yet',
                          textAlign: TextAlign.center,
                          style: TextStyle(fontSize: 20),
                        ),
                        SizedBox(height: 8),
                        Text(
                          'Complete a scan to save its result here.',
                          textAlign: TextAlign.center,
                        ),
                      ],
                    )
                  : ListView.builder(
                      padding: const EdgeInsets.fromLTRB(16, 0, 16, 20),
                      itemCount: records.length,
                      itemBuilder: (context, index) {
                        final record = records[index];
                        return Card(
                          child: ListTile(
                            key: ValueKey('history_${record.id}'),
                            leading: Icon(
                              record.result.hasPrediction
                                  ? Icons.analytics_outlined
                                  : Icons.help_outline,
                            ),
                            title: Text(
                              record.result.filename,
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                            ),
                            subtitle: Text(
                              '${record.title}\n'
                              '${formatScanDate(record.createdAtUtc)}',
                            ),
                            isThreeLine: true,
                            trailing: const Icon(Icons.chevron_right),
                            onTap: _clearing
                                ? null
                                : () {
                                    Navigator.of(context).push(
                                      MaterialPageRoute<void>(
                                        builder: (context) =>
                                            ScanHistoryDetailScreen(
                                              record: record,
                                              store: _store,
                                            ),
                                      ),
                                    );
                                  },
                          ),
                        );
                      },
                    ),
            ),
          ],
        );
      },
    );
  }
}
