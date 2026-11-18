import 'dart:async';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import '../models/analysis_result.dart';
import '../models/scan_record.dart';
import '../services/scan_history_store.dart';
import '../services/video_upload_service.dart';
import 'analysis_result_card.dart';

typedef VideoAnalyser = Future<AnalysisResult> Function(
  PlatformFile file, {
  required Future<void> abortTrigger,
});

class VideoUploadCard extends StatefulWidget {
  final PlatformFile file;
  final ValueChanged<bool>? onBusyChanged;
  final Future<void> Function()? beforeAnalyse;
  final ScanHistoryStore? historyStore;
  final VideoAnalyser? analyser;
  const VideoUploadCard({
    super.key,
    required this.file,
    this.onBusyChanged,
    this.beforeAnalyse,
    this.historyStore,
    this.analyser,
  });
  @override
  State<VideoUploadCard> createState() => _VideoUploadCardState();
}

class _VideoUploadCardState extends State<VideoUploadCard> {
  bool _uploading = false;
  AnalysisResult? _result;
  String? _error;
  Completer<void>? _abort;
  Timer? _timer;
  final Stopwatch _elapsed = Stopwatch();
  ScanRecord? _pendingRecord;
  bool _savingHistory = false;
  bool _historySaved = false;
  String? _historyError;

  Future<void> _saveHistory() async {
    final record = _pendingRecord;
    if (record == null || _savingHistory) {
      return;
    }
    setState(() {
      _savingHistory = true;
      _historyError = null;
    });
    try {
      await (widget.historyStore ?? scanHistoryStore).save(record);
      if (mounted) {
        setState(() => _historySaved = true);
      }
    } catch (error) {
      debugPrint('History save error: $error');
      if (mounted) {
        setState(
          () => _historyError =
              'This result could not be saved to '
              'history. The analysis result is still available below.',
        );
      }
    } finally {
      if (mounted) {
        setState(() => _savingHistory = false);
      }
    }
  }

  void _abortWaiting() {
    final cancellation = _abort;
    if (cancellation != null && !cancellation.isCompleted) {
      cancellation.complete();
    }
  }

  Future<void> _upload() async {
    if (_uploading || _savingHistory || !mounted) {
      return;
    }
    final cancellation = Completer<void>();
    _abort = cancellation;
    _elapsed
      ..reset()
      ..start();
    setState(() {
      _uploading = true;
      _result = null;
      _error = null;
      _pendingRecord = null;
      _historySaved = false;
      _historyError = null;
    });
    widget.onBusyChanged?.call(true);
    _timer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) {
        setState(() {});
      }
    });
    AnalysisResult? result;
    String? errorMessage;
    try {
      await widget.beforeAnalyse?.call();
      if (!mounted) {
        return;
      }
      if (cancellation.isCompleted) {
        errorMessage = 'Stopped waiting for this result.';
        return;
      }
      result = await (widget.analyser ?? VideoUploadService.upload)(
        widget.file,
        abortTrigger: cancellation.future,
      );
      if (cancellation.isCompleted || !mounted) {
        result = null;
        errorMessage = 'Stopped waiting for this result.';
      }
    } on TimeoutException {
      errorMessage = 'The request timed out. The server may still be finishing this scan. Wait a moment, check the connection, then retry.';
    } on http.RequestAbortedException {
      errorMessage = 'Stopped waiting for this result. The server may finish the current scan before accepting another video.';
    } on UploadException catch (error) {
      errorMessage = error.message;
    } on FormatException {
      errorMessage = 'The service returned an incompatible response. Check that both the app and backend have the integration update.';
    } catch (error) {
      debugPrint('Video analysis error: $error');
      errorMessage = 'The result could not be received. Check the connection and try again.';
    } finally {
      _timer?.cancel();
      _elapsed.stop();
      _abortWaiting();
      _abort = null;
      if (mounted) {
        setState(() {
          _uploading = false;
          _result = result;
          _error = errorMessage;
        });
        widget.onBusyChanged?.call(false);
      }
    }
    if (result != null && mounted) {
      _pendingRecord = ScanRecord(createdAt: DateTime.now(), result: result);
      await _saveHistory();
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    _elapsed.stop();
    _abortWaiting();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final result = _result;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Analyse video',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 8),
                const Text(
                  'Send this clip to the analysis service. A scan can take a minute or more on a laptop CPU.',
                ),
                if (_error != null) ...[
                  const SizedBox(height: 12),
                  Text(
                    _error!,
                    style: TextStyle(
                      color: Theme.of(context).colorScheme.error,
                    ),
                  ),
                ],
                const SizedBox(height: 16),
                FilledButton.icon(
                  onPressed: _uploading || _savingHistory ? null : _upload,
                  icon: const Icon(Icons.analytics_outlined),
                  label: Text(
                    _uploading
                        ? 'Uploading and analysing...'
                        : result != null
                        ? 'Analyse again'
                        : 'Analyse video',
                  ),
                ),
                if (_uploading) ...[
                  const SizedBox(height: 12),
                  const LinearProgressIndicator(),
                  const SizedBox(height: 8),
                  Text('Elapsed: ${_elapsed.elapsed.inSeconds} seconds'),
                  const Text('Keep the app open and the service connected.'),
                  TextButton(
                    onPressed: _abortWaiting,
                    child: const Text('Stop waiting'),
                  ),
                ],
              ],
            ),
          ),
        ),
        if (result != null) AnalysisResultCard(result: result),
        if (result != null)
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 8, 12, 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (_savingHistory) ...[
                  const LinearProgressIndicator(),
                  const SizedBox(height: 8),
                  const Text('Saving result to local history...'),
                ] else if (_historySaved) ...[
                  const Text('Saved to History on this device.'),
                ] else if (_historyError != null) ...[
                  Text(_historyError!),
                  TextButton.icon(
                    onPressed: _saveHistory,
                    icon: const Icon(Icons.save_outlined),
                    label: const Text('Retry saving'),
                  ),
                ],
              ],
            ),
          ),
      ],
    );
  }
}
