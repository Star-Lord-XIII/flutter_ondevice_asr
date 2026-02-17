import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_ondevice_asr/audio.dart';

void main() {
  test('load audio', () async {
    WidgetsFlutterBinding.ensureInitialized();
    final testAudioFloat32List = await Audio.instance.loadAudio('assets/audio/crisp_autumn.wav');
    expect(testAudioFloat32List.runtimeType.toString(), "Float32List");
    expect(testAudioFloat32List.length, 61440);
  });
}
