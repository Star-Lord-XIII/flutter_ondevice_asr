import 'package:flutter/material.dart';
import 'package:flutter_ondevice_asr/util/audio.dart';
import 'package:flutter_test/flutter_test.dart';

import 'test_utils.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  final testAudioFile = toAbsolutePath('assets/audio/crisp_autumn.wav');

  debugPrint('Unit test paths (absolute, filesystem-based):');
  debugPrint('  testAudioFile: $testAudioFile');

  test('load audio', () async {
    final testAudioFloat32List = await Audio.instance.loadAudio(testAudioFile);
    expect(testAudioFloat32List.runtimeType.toString(), "Float32List");
    expect(testAudioFloat32List.length, 61440);
  });
}
