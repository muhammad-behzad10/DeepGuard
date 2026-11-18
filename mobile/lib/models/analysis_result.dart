class AnalysisResult {
  final String uploadId, filename, sha256, status;
  final int sizeBytes, sampledFrames, usableFaces, minimumFaces;
  final String? label, reason;
  final double? fakeScore;
  final Map<String, int> frameCounts;
  final double processingSeconds;
  final String apiVersion, modelId, checkpointSha256, preprocessingVersion;
  final double? decisionThreshold;
  const AnalysisResult({
    required this.uploadId,
    required this.filename,
    required this.sha256,
    required this.status,
    required this.sizeBytes,
    required this.sampledFrames,
    required this.usableFaces,
    required this.minimumFaces,
    required this.label,
    required this.reason,
    required this.fakeScore,
    required this.frameCounts,
    required this.processingSeconds,
    required this.apiVersion,
    required this.modelId,
    required this.checkpointSha256,
    required this.preprocessingVersion,
    required this.decisionThreshold,
  });
  bool get hasPrediction => status == 'analysed';

  factory AnalysisResult.fromJson(Object? data, {required int expectedBytes}) {
    const bad = FormatException('Unexpected analysis response.');
    if (data is! Map<String, dynamic> ||
        data['api_version'] != '0.6.0' ||
        data['upload_id'] is! String ||
        (data['upload_id'] as String).isEmpty ||
        data['filename'] is! String ||
        data['size_bytes'] is! int ||
        data['size_bytes'] != expectedBytes ||
        expectedBytes <= 0 ||
        data['sha256'] is! String ||
        !RegExp(r'^[0-9a-f]{64}$').hasMatch(data['sha256'] as String) ||
        data['stored'] != false) {
      throw bad;
    }
    final prep = data['preprocessing'];
    final timing = data['timing_seconds'];
    final model = data['model'];
    if (prep is! Map<String, dynamic> ||
        timing is! Map<String, dynamic> ||
        model is! Map<String, dynamic> ||
        model['id'] is! String ||
        (model['id'] as String).isEmpty ||
        model['checkpoint_sha256'] is! String ||
        !RegExp(r'^[0-9a-f]{64}$')
            .hasMatch(model['checkpoint_sha256'] as String) ||
        prep['version'] is! String ||
        (prep['version'] as String).isEmpty) {
      throw bad;
    }
    final sampled = prep['sampled_frames'];
    final usable = prep['usable_faces'];
    final minimum = prep['minimum_usable_faces'];
    final seconds = timing['processing'];
    final counts = prep['counts'];
    if (sampled is! int ||
        sampled < 1 ||
        sampled > 16 ||
        usable is! int ||
        usable < 0 ||
        usable > sampled ||
        minimum != 8 ||
        seconds is! num ||
        !seconds.isFinite ||
        seconds < 0 ||
        counts is! Map<String, dynamic>) {
      throw bad;
    }
    final frameCounts = <String, int>{};
    for (final entry in counts.entries) {
      final value = entry.value;
      if (value is! int || value < 0) {
        throw bad;
      }
      frameCounts[entry.key] = value;
    }
    if (frameCounts.values.fold<int>(0, (a, b) => a + b) != sampled ||
        (frameCounts['accepted'] ?? 0) != usable) {
      throw bad;
    }
    String? label, reason;
    double? fakeScore;
    final status = data['status'];
    final prediction = data['prediction'];
    if (status == 'analysed') {
      if (data['analysis_performed'] != true ||
          usable < 8 ||
          prediction is! Map<String, dynamic> ||
          prediction['calibrated'] != false ||
          prediction['decision_threshold'] != 0.5 ||
          prediction['frames_scored'] != usable) {
        throw bad;
      }
      final score = prediction['fake_score'];
      final realScore = prediction['real_score'];
      if (score is! num ||
          !score.isFinite ||
          score < 0 ||
          score > 1 ||
          realScore is! num ||
          !realScore.isFinite ||
          (score + realScore - 1).abs() > 0.000001) {
        throw bad;
      }
      fakeScore = score.toDouble();
      label = fakeScore >= 0.5 ? 'Fake' : 'Real';
      if (prediction['label'] != label ||
          prediction['class_id'] != (label == 'Fake' ? 1 : 0)) {
        throw bad;
      }
    } else if (status == 'insufficient_evidence') {
      if (data['analysis_performed'] != false ||
          prediction != null ||
          usable >= 8 ||
          data['reason'] is! String ||
          (data['reason'] as String).isEmpty) {
        throw bad;
      }
      reason = data['reason'] as String;
    } else {
      throw bad;
    }
    return AnalysisResult(
      uploadId: data['upload_id'] as String,
      filename: data['filename'] as String,
      sha256: data['sha256'] as String,
      status: status as String,
      sizeBytes: data['size_bytes'] as int,
      sampledFrames: sampled,
      usableFaces: usable,
      minimumFaces: minimum as int,
      label: label,
      reason: reason,
      fakeScore: fakeScore,
      frameCounts: Map.unmodifiable(frameCounts),
      processingSeconds: seconds.toDouble(),
      apiVersion: data['api_version'] as String,
      modelId: model['id'] as String,
      checkpointSha256: model['checkpoint_sha256'] as String,
      preprocessingVersion: prep['version'] as String,
      decisionThreshold: prediction is Map<String, dynamic>
          ? (prediction['decision_threshold'] as num).toDouble()
          : null,
    );
  }
}
