import 'dart:io';
import 'dart:math';
import 'package:flutter/services.dart';
import 'package:flutter_ondevice_asr/common/result.dart';
import 'package:flutter_ondevice_asr/model/transcription_result.dart';
import 'package:flutter_ondevice_asr/util/audio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:flutter_ondevice_asr/flutter_ondevice_asr.dart';
import 'package:logging/logging.dart';
import 'package:path_provider/path_provider.dart';


void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const testAudioFile = 'packages/flutter_ondevice_asr/assets/audio/jfk_asknot.wav';
  const expectedTranscript = 'And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.';
  const modelDirectory = 'assets/transcribers/whisper/models/whisper_tiny/default_int8';
  const String language = 'en';

  // Alternative test audio:
  // const testAudioFile = 'packages/flutter_ondevice_asr/assets/audio/crisp_autumn.wav';
  // const expectedTranscript = 'crisp autumn leaves crunch underfoot';
  // Alternative: use external model paths (not bundled):
  // const modelDirectory = '/tmp/onnx_tiny/default';  // multilingual
  // const modelDirectory = '/tmp/onnx_tiny_en/default';  // English-only

  setUp(() {
    Logger.root.level = Level.ALL; // defaults to Level.INFO
    Logger.root.onRecord.listen((record) {
      print('${record.loggerName}: ${record.time}: ${record.message}');
    });
  });

  testWidgets('transcribe test audio', (WidgetTester tester) async {
    // 1. Initialize stopwatch to measure durations
    final totalSw = Stopwatch()..start();
    final stepSw = Stopwatch();

    void logStep(String message) {
      print('[${DateTime.now()}] $message (${stepSw.elapsedMilliseconds}ms)');
      stepSw.reset();
      stepSw.start();
    }

    print('[${DateTime.now()}] START TEST');
    stepSw.start();

    final whisper = Transcriber.getInstance(TranscriberType.whisper);

    // 2. Load models
    final loadModelResult = await whisper.loadModel(modelDirectory: modelDirectory, languageCode: language);
    expect(loadModelResult is Ok, true);
    logStep('Models loaded');

    // 3. Load test audio
    final audioAsset = await rootBundle.load(testAudioFile);
    final tempDir = await getTemporaryDirectory();

    // Ensure the temporary directory exists
    if (!await tempDir.exists()) {
      await tempDir.create(recursive: true);
    }

    final tempAudioFile = File('${tempDir.path}/test_audio.wav');
    await tempAudioFile.writeAsBytes(audioAsset.buffer.asUint8List());

    final testAudioFloat32List = await Audio.instance.loadAudio(tempAudioFile.path);
    logStep('Audio file prepared and loaded');

    // Audio duration
    final audioDurationSec = testAudioFloat32List.length / 16000.0;
    print('Audio duration: ${audioDurationSec.toStringAsFixed(2)} seconds (${testAudioFloat32List.length} samples @ 16kHz)');

    // 4. Transcribe (5 runs for statistics)
    print('\n=== Running transcription 5 times for performance statistics ===');
    final durations = <double>[];
    String? transcript;

    for (int run = 0; run < 5; run++) {
      print('\n--- Run ${run + 1}/5 ---');
      stepSw.reset();
      stepSw.start();
      final result = await whisper.transcribe(testAudioFloat32List) as Ok<TranscriptionResult>;
      final durationMs = stepSw.elapsedMilliseconds.toDouble();
      durations.add(durationMs);
      print('Duration: ${durationMs.toStringAsFixed(1)} ms');
      if (run == 0) transcript = result.value.text;
    }

    // Calculate average and standard deviation
    final avg = durations.reduce((a, b) => a + b) / durations.length;
    final variance = durations.map((d) => pow(d - avg, 2)).reduce((a, b) => a + b) / durations.length;
    final std = sqrt(variance);

    print('\n=== Performance Statistics ===');
    print('Average: ${avg.toStringAsFixed(1)} ms');
    print('Std Dev: ${std.toStringAsFixed(1)} ms');
    print('Min: ${durations.reduce(min).toStringAsFixed(1)} ms');
    print('Max: ${durations.reduce(max).toStringAsFixed(1)} ms');
    print('\nTranscript: $transcript');

    // 5. Clean up & Verify
    await tempAudioFile.delete();
    expect(transcript, expectedTranscript);

    print('[${DateTime.now()}] TEST PASSED - Total duration: ${totalSw.elapsed.inSeconds}s');
  });
}
