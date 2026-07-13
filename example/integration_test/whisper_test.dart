import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:logging/logging.dart';

import 'whisper_non_streaming_test_util.dart';

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const testAudioFile =
      'packages/flutter_ondevice_asr/assets/audio/jfk_asknot.wav';
  const expectedTranscript =
      'And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.';
  const modelDirectory =
      'assets/transcribers/whisper/models/whisper_tiny/default_int8';
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
    const collectPerformance = bool.fromEnvironment('PERFORMANCE', defaultValue: false);

    if (collectPerformance) {
      await binding.traceAction(() async {
        await WhisperNonStreamingTestUtil.testNonStreamingTranscription(
            testAudioFile: testAudioFile,
            expectedTranscript: expectedTranscript,
            modelDirectory: modelDirectory,
            language: language);
      }, streams: ["Dart"]);
    } else {
      await WhisperNonStreamingTestUtil.testNonStreamingTranscription(
          testAudioFile: testAudioFile,
          expectedTranscript: expectedTranscript,
          modelDirectory: modelDirectory,
          language: language);
    }
  });
}
