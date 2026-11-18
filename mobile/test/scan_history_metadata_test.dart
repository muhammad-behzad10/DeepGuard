import 'dart:convert';
import 'dart:io';

import 'package:deepguard/models/analysis_result.dart';
import 'package:flutter_test/flutter_test.dart';

Map<String, dynamic> response(String name) =>
    jsonDecode(File('test/fixtures/${name}_api.json').readAsStringSync())
        as Map<String, dynamic>;

void main() {
  test(
    'both completed response types retain model and preprocessing metadata',
    () {
      for (final name in ['single', 'blank']) {
        final data = response(name);
        final r = AnalysisResult.fromJson(
          data,
          expectedBytes: data['size_bytes'] as int,
        );
        expect(r.modelId, data['model']['id']);
        expect(r.checkpointSha256, data['model']['checkpoint_sha256']);
        expect(r.preprocessingVersion, data['preprocessing']['version']);
        expect(r.decisionThreshold, name == 'single' ? 0.5 : isNull);
      }
    },
  );

  test(
    'missing provenance is rejected instead of saving invented metadata',
    () {
      for (final field in ['id', 'checkpoint_sha256']) {
        final data = response('single');
        (data['model'] as Map<String, dynamic>).remove(field);
        expect(
          () => AnalysisResult.fromJson(
            data,
            expectedBytes: data['size_bytes'] as int,
          ),
          throwsFormatException,
        );
      }
      final data = response('single');
      data['preprocessing']['version'] = '';
      expect(
        () => AnalysisResult.fromJson(
          data,
          expectedBytes: data['size_bytes'] as int,
        ),
        throwsFormatException,
      );
    },
  );
}
