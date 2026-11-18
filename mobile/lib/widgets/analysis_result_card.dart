import 'package:flutter/material.dart';

import '../models/analysis_result.dart';

class AnalysisResultCard extends StatelessWidget {
  final AnalysisResult result;
  const AnalysisResultCard({super.key, required this.result});
  @override
  Widget build(BuildContext context) {
    final color = !result.hasPrediction
        ? const Color(0xFF475569)
        : result.label == 'Fake'
        ? const Color(0xFF9A3412)
        : const Color(0xFF0F766E);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              result.hasPrediction
                  ? 'Prediction: ${result.label}'
                  : 'Insufficient evidence',
              style: TextStyle(
                fontSize: 23,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
            const SizedBox(height: 10),
            Text(result.filename, maxLines: 2, overflow: TextOverflow.ellipsis),
            const SizedBox(height: 12),
            if (result.hasPrediction) ...[
              Text(
                'Fake model score: ${result.fakeScore!.toStringAsFixed(4)} / 1.0',
              ),
              const SizedBox(height: 6),
              const Text(
                'This score is not a calibrated probability or an accuracy percentage.',
              ),
              const SizedBox(height: 8),
              const Text(
                'Research prototype: predictions can be wrong. This assessment is based on sampled face frames.',
              ),
            ] else ...[
              Text(result.reason!),
              if ((result.frameCounts['multiple_faces'] ?? 0) > 0)
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Text(
                    'Multiple faces detected in ${result.frameCounts['multiple_faces']} sampled frames.',
                  ),
                ),
            ],
            const SizedBox(height: 12),
            Text(
              'Usable face frames: ${result.usableFaces}/${result.sampledFrames}',
            ),
            Text(
              'Processing: ${result.processingSeconds.toStringAsFixed(1)} seconds',
            ),
            const SizedBox(height: 12),
            const Text(
              'Temporary video and crops were removed from the server. '
              'Local history stores result details only.',
            ),
          ],
        ),
      ),
    );
  }
}
