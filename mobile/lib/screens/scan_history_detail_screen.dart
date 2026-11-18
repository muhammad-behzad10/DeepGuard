import 'package:flutter/material.dart';

import '../models/scan_record.dart';
import '../services/scan_history_store.dart';
import '../widgets/analysis_result_card.dart';

class ScanHistoryDetailScreen extends StatefulWidget {
  final ScanRecord record;
  final ScanHistoryStore store;
  const ScanHistoryDetailScreen({
    super.key,
    required this.record,
    required this.store,
  });

  @override
  State<ScanHistoryDetailScreen> createState() =>
      _ScanHistoryDetailScreenState();
}

class _ScanHistoryDetailScreenState extends State<ScanHistoryDetailScreen> {
  bool _deleting = false;

  Future<void> _delete() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete saved result?'),
        content: const Text(
          'This removes the saved result from history. '
          'It does not delete your original video.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) {
      return;
    }
    setState(() => _deleting = true);
    try {
      await widget.store.delete(widget.record.id);
      if (mounted) {
        Navigator.of(context).pop();
      }
    } catch (_) {
      if (mounted) {
        setState(() => _deleting = false);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('The saved result could not be deleted. Try again.'),
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final record = widget.record;
    final result = record.result;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Saved scan'),
        actions: [
          IconButton(
            onPressed: _deleting ? null : _delete,
            tooltip: 'Delete saved result',
            icon: const Icon(Icons.delete_outline),
          ),
        ],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            if (_deleting) const LinearProgressIndicator(),
            Text('Saved ${formatScanDate(record.createdAtUtc)} (local time)'),
            const SizedBox(height: 8),
            const Text('This is a saved result, not a new analysis.'),
            const SizedBox(height: 12),
            AnalysisResultCard(result: result),
            const SizedBox(height: 12),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Scan details',
                      style: Theme.of(context).textTheme.titleLarge,
                    ),
                    const SizedBox(height: 12),
                    SelectableText('Scan ID: ${record.id}'),
                    const SizedBox(height: 8),
                    SelectableText('Model: ${result.modelId}'),
                    const SizedBox(height: 8),
                    SelectableText(
                      'Preprocessing: ${result.preprocessingVersion}',
                    ),
                    const SizedBox(height: 8),
                    Text(
                      result.decisionThreshold == null
                          ? 'Decision threshold: no prediction issued'
                          : 'Decision threshold: ${result.decisionThreshold}',
                    ),
                    const SizedBox(height: 8),
                    Text('API version: ${result.apiVersion}'),
                    const SizedBox(height: 8),
                    SelectableText('Model SHA-256: ${result.checkpointSha256}'),
                    const SizedBox(height: 8),
                    SelectableText('Video SHA-256: ${result.sha256}'),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),
            const Text(
              'History stores result metadata locally. It does not retain '
              'videos or face crops. Clearing app storage or uninstalling the '
              'app can remove your saved history.',
            ),
          ],
        ),
      ),
    );
  }
}
