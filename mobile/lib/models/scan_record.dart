import 'dart:convert';

import 'analysis_result.dart';

/// One accepted server response. No video bytes, crops or local media paths.
class ScanRecord {
  final DateTime createdAtUtc;
  final AnalysisResult result;

  ScanRecord({required DateTime createdAt, required this.result})
    : createdAtUtc = createdAt.toUtc();

  String get id => result.uploadId;

  String get title => result.hasPrediction
      ? 'Prediction: ${result.label}'
      : 'Insufficient evidence';

  Map<String, Object?> toDatabase() => {
    'scan_id': id,
    'created_at_utc': createdAtUtc.toIso8601String(),
    'filename': result.filename,
    'input_sha256': result.sha256,
    'size_bytes': result.sizeBytes,
    'status': result.status,
    'label': result.label,
    'fake_score': result.fakeScore,
    'decision_threshold': result.decisionThreshold,
    'reason': result.reason,
    'api_version': result.apiVersion,
    'model_id': result.modelId,
    'checkpoint_sha256': result.checkpointSha256,
    'preprocessing_version': result.preprocessingVersion,
    'sampled_frames': result.sampledFrames,
    'usable_faces': result.usableFaces,
    'minimum_faces': result.minimumFaces,
    'processing_seconds': result.processingSeconds,
    'frame_counts_json': jsonEncode(result.frameCounts),
    'schema_version': 1,
  };

  factory ScanRecord.fromDatabase(Map<String, Object?> row) {
    if (row['schema_version'] != 1) {
      throw const FormatException('Unsupported history record version.');
    }
    final rawCounts = jsonDecode(row['frame_counts_json'] as String);
    if (rawCounts is! Map<String, dynamic>) {
      throw const FormatException('Invalid saved frame counts.');
    }
    final counts = rawCounts.map((key, value) => MapEntry(key, value as int));
    // Read the stored decision: do not reinterpret an old scan using today's
    // API parser, model, threshold or preprocessing settings.
    return ScanRecord(
      createdAt: DateTime.parse(row['created_at_utc'] as String),
      result: AnalysisResult(
        uploadId: row['scan_id'] as String,
        filename: row['filename'] as String,
        sha256: row['input_sha256'] as String,
        sizeBytes: row['size_bytes'] as int,
        status: row['status'] as String,
        label: row['label'] as String?,
        fakeScore: (row['fake_score'] as num?)?.toDouble(),
        decisionThreshold: (row['decision_threshold'] as num?)?.toDouble(),
        reason: row['reason'] as String?,
        apiVersion: row['api_version'] as String,
        modelId: row['model_id'] as String,
        checkpointSha256: row['checkpoint_sha256'] as String,
        preprocessingVersion: row['preprocessing_version'] as String,
        sampledFrames: row['sampled_frames'] as int,
        usableFaces: row['usable_faces'] as int,
        minimumFaces: row['minimum_faces'] as int,
        processingSeconds: (row['processing_seconds'] as num).toDouble(),
        frameCounts: Map.unmodifiable(counts),
      ),
    );
  }
}

String formatScanDate(DateTime utc) {
  final d = utc.toLocal();
  String two(int n) => n.toString().padLeft(2, '0');
  return '${d.year}-${two(d.month)}-${two(d.day)} '
      '${two(d.hour)}:${two(d.minute)}:${two(d.second)}';
}
