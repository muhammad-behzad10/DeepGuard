import 'dart:convert';
import 'dart:io';

import 'package:deepguard/models/analysis_result.dart';
import 'package:deepguard/widgets/analysis_result_card.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

// API fixtures use synthetic videos; these tests do not measure model accuracy.
Map<String, dynamic> fixture(String name) =>
    jsonDecode(File('test/fixtures/${name}_api.json').readAsStringSync())
        as Map<String, dynamic>;
AnalysisResult parse(Map<String, dynamic> data) =>
    AnalysisResult.fromJson(data, expectedBytes: data['size_bytes'] as int);
void main() {
  test('actual scored API response is accepted', () {
    final r = parse(fixture('single'));
    expect(r.hasPrediction, isTrue);
    expect(r.label, 'Fake');
    expect(r.usableFaces, 16);
  });
  test('no-face response has no classification or score', () {
    final r = parse(fixture('blank'));
    expect(r.hasPrediction, isFalse);
    expect(r.label, isNull);
    expect(r.fakeScore, isNull);
  });
  test('wrong label, invalid score and old receipt are rejected', () {
    var d = fixture('single');
    d['prediction']['label'] = 'Real';
    expect(() => parse(d), throwsFormatException);
    d = fixture('single');
    d['prediction']['fake_score'] = double.nan;
    expect(() => parse(d), throwsFormatException);
    d = fixture('single');
    d['status'] = 'received';
    expect(() => parse(d), throwsFormatException);
  });
  test('wrong upload size and inconsistent counts are rejected', () {
    final d = fixture('single');
    expect(
      () => AnalysisResult.fromJson(d, expectedBytes: 1),
      throwsFormatException,
    );
    d['preprocessing']['usable_faces'] = 7;
    expect(() => parse(d), throwsFormatException);
  });
  testWidgets('Real/Fake cards display the model score with limitations', (
    tester,
  ) async {
    for (final label in ['Fake', 'Real']) {
      final d = fixture('single');
      if (label == 'Real') {
        d['prediction']['label'] = 'Real';
        d['prediction']['class_id'] = 0;
        d['prediction']['fake_score'] = 0.2;
        d['prediction']['real_score'] = 0.8;
      }
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: SingleChildScrollView(
              child: AnalysisResultCard(result: parse(d)),
            ),
          ),
        ),
      );
      expect(find.text('Prediction: $label'), findsOneWidget);
      expect(find.textContaining('Fake model score:'), findsOneWidget);
      expect(
        find.textContaining('not a calibrated probability'),
        findsOneWidget,
      );
      expect(tester.takeException(), isNull);
    }
  });
  testWidgets('insufficient evidence is not presented as Real', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: AnalysisResultCard(result: parse(fixture('blank'))),
          ),
        ),
      ),
    );
    expect(find.text('Insufficient evidence'), findsOneWidget);
    expect(find.textContaining('Prediction:'), findsNothing);
    expect(find.textContaining('Fake model score:'), findsNothing);
    expect(tester.takeException(), isNull);
  });
  testWidgets('result fits a narrow phone screen', (tester) async {
    tester.view.physicalSize = const Size(320, 640);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: AnalysisResultCard(result: parse(fixture('single'))),
          ),
        ),
      ),
    );
    expect(tester.takeException(), isNull);
  });
}
