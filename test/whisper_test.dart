import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_ondevice_asr/common/result.dart';
import 'package:flutter_ondevice_asr/model/transcription_result.dart';
import 'package:flutter_ondevice_asr/models/whisper/whisper_transcriber.dart';
import 'package:flutter_ondevice_asr/util/audio.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  // Model configuration - using bundled multilingual model
  // const modelDirectory = 'assets/transcribers/whisper/models/whisper_tiny/default';
  const modelDirectory = 'assets/transcribers/whisper/models/whisper_tiny/default_int8';
  // const modelDirectory = '/Users/chintanghate/Projects/ktomanek/flutter_ondevice_asr/assets/transcribers/whisper/models/whisper_tiny/default_int8';
  const language = 'en';

  const testAudioFile = 'assets/audio/jfk_asknot.wav';
  const expectedTranscript =
      'And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.';

  // // Alternative test audio:
  // const testAudioFile = 'assets/audio/crisp_autumn.wav';
  // const expectedTranscript = 'Crisp, autumn leaves crunch, underfoot.';

  test(
      'language validation',
          () async {
        // Test 2 supported languages - should not throw
        expect(
              () => WhisperTranscriber().loadModel(
            modelDirectory: modelDirectory,
            languageCode: 'en',
          ),
          returnsNormally,
        );

        expect(
              () => WhisperTranscriber().loadModel(
            modelDirectory: modelDirectory,
            languageCode: 'de',
          ),
          returnsNormally,
        );

        // Test 1 unsupported language - should throw ArgumentError
        final result = await WhisperTranscriber().loadModel(
          modelDirectory: modelDirectory,
          languageCode: 'xyz',
        );
        expect(result is Error, true);
        if (result is Error) {
          expect(
            result.error.toString(),
            contains('Language "xyz" not found in model configuration.'),
          );
        }
      });

  test(
      'transcribe test audio',
          () async {
        final whisper = WhisperTranscriber();
        await whisper.loadModel(
          modelDirectory: modelDirectory,
          languageCode: language,
        );
        final audioData = await Audio.instance.loadAudio(testAudioFile);

        // Audio is sampled at 16kHz (16000 samples per second)
        final audioDurationSec = audioData.length / 16000.0;
        debugPrint(
          '\nAudio duration: ${audioDurationSec.toStringAsFixed(2)} seconds (${audioData.length} samples @ 16kHz)',
        );

        // Test 1: WITHOUT confidence (run 5 times for statistics)
        debugPrint('\n=== Test 1: transcribe() WITHOUT confidence (5 runs) ===');

        String? transcript1;
        final durations = <double>[];

        for (int run = 0; run < 5; run++) {
          debugPrint('\n--- Run ${run + 1}/5 ---');
          final stopwatch = Stopwatch()..start();
          final result = await whisper.transcribe(audioData, maxOutputTokens: 32);
          stopwatch.stop();
          final durationMs = stopwatch.elapsedMilliseconds.toDouble();
          durations.add(durationMs);
          debugPrint('Duration: ${durationMs.toStringAsFixed(1)} ms');
          if (run == 0)
            transcript1 = (result as Ok<TranscriptionResult>).value.text;
        }

        // Calculate average and standard deviation
        final avg = durations.reduce((a, b) => a + b) / durations.length;
        final variance =
            durations.map((d) => pow(d - avg, 2)).reduce((a, b) => a + b) /
                durations.length;
        final std = sqrt(variance);

        debugPrint('\n=== Performance Statistics ===');
        debugPrint('Average: ${avg.toStringAsFixed(1)} ms');
        debugPrint('Std Dev: ${std.toStringAsFixed(1)} ms');
        debugPrint('Min: ${durations.reduce(min).toStringAsFixed(1)} ms');
        debugPrint('Max: ${durations.reduce(max).toStringAsFixed(1)} ms');
        debugPrint('\nTranscript: $transcript1');

        // Test 2: WITH confidence (single run)
        debugPrint('\n=== Test 2: transcribe() WITH confidence (1 run) ===');
        final result = await whisper.transcribe(
          audioData,
          getWordDetails: true,
          maxOutputTokens: 32,
        );
        final transcript2 = (result as Ok<TranscriptionResult>).value.text;
        final words = result.value.words!;

        debugPrint('Transcript: $transcript2');
        debugPrint(
          'Word confidences: ${words.map((w) => '${w.word}(${w.confidence.toStringAsFixed(3)})').join(', ')}',
        );
        debugPrint('Number of words: ${words.length}');

        expect(transcript1, expectedTranscript);
        expect(transcript2, expectedTranscript);
      });
}
