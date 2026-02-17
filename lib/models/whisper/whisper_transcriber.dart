import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_ondevice_asr/audio.dart';
import 'package:flutter_ondevice_asr/onnx_config.dart';
import 'package:flutter_ondevice_asr/transcriber.dart';
import 'package:flutter_ondevice_asr/transcription_result.dart';
import 'package:flutter_ondevice_asr/utils.dart';
import 'package:flutter_ondevice_asr/models/whisper/whisper_tokenizer.dart';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';

/// Whisper ASR transcriber implementation.
class WhisperTranscriber implements Transcriber {
  // Default tokens per second for dynamic calculation (conservative estimate for fast speech)
  static const double defaultTokensPerSecond = 6.0;

  OrtSession? superEncoderSession;  // Combined preprocessor + encoder
  OrtSession? decoderSession;
  OrtSession? decoderWithPastSession;
  final String modelDirectory;
  final String? language;
  final double tokensPerSecond;
  final bool verbose;

  @override
  final OnnxConfig onnxConfig;

  // Token IDs loaded from model config
  late int sotToken;
  late int eotToken;
  late int noTimestampsToken;
  int? transcribeToken;
  int? languageToken;

  /// Create a Whisper transcriber instance
  ///
  /// Parameters:
  /// - [modelDirectory]: Path to the Whisper ONNX model directory
  /// - [language]: Language code from getAllSupportedLanguages().
  /// - [tokensPerSecond]: Multiplier for dynamic maxOutputTokens calculation.
  ///   When maxOutputTokens is not explicitly provided, calculate dynamically:
  ///   `audioDuration * tokensPerSecond` (clamped 10-224).
  ///   Default: 6.0 tokens/sec (conservative for fast speech and prevents hallucinations).
  ///   Increase if transcriptions are getting cut off, decrease to save computation.
  /// - [onnxConfig]: ONNX Runtime configuration (threading, optimization, execution providers).
  ///   If not provided, uses default configuration optimized for mobile devices.
  /// - [verbose]: Enable verbose logging for debugging (default: false)
  ///
  /// Throws [ArgumentError] if the specified language is not supported.
  WhisperTranscriber({
    this.modelDirectory = 'assets/models/whisper_tiny_multilingual/default',
    this.language = 'en',
    this.tokensPerSecond = defaultTokensPerSecond,
    OnnxConfig? onnxConfig,
    this.verbose = false,
  }) : onnxConfig = onnxConfig ?? const TranscriberOnnxConfig() {
    // Validate language is supported
    if (language != null && !getAllSupportedLanguages().contains(language)) {
      throw ArgumentError(
        'Unsupported language: "$language". '
        'Supported languages: ${getAllSupportedLanguages().join(", ")}. '
        'Use WhisperTranscriber.getAllSupportedLanguages() to get the full list.',
      );
    }
  }


  /// Helper to determine if path is external (absolute file path) vs bundled asset
  bool _isExternalPath(String path) {
    return path.startsWith('/') || path.startsWith('file://');
  }

  /// Load bytes from either external file or bundled asset
  Future<Uint8List> _loadBytes(String path) async {
    if (_isExternalPath(path)) {
      debugPrint("Loading from external file: $path");
      return await File(path).readAsBytes();
    } else {
      final assetPath = '${isRunningInTestEnvironment() ? '' : 'packages/flutter_ondevice_asr/'}$path';
      debugPrint("Loading from asset: $assetPath");
      final data = await rootBundle.load(assetPath);
      return data.buffer.asUint8List();
    }
  }

  /// Load string from either external file or bundled asset
  Future<String> _loadString(String path) async {
    if (_isExternalPath(path)) {
      return await File(path).readAsString();
    } else {
      final assetPath = '${isRunningInTestEnvironment() ? '' : 'packages/flutter_ondevice_asr/'}$path';
      return await rootBundle.loadString(assetPath);
    }
  }

  @override
  Future<void> loadModels() async {

    // Load tokenizer vocab based on model type
    final vocabFileName = language != null ? 'vocab_multilingual.json' : 'vocab_en.json';
    final vocabPath = '${isRunningInTestEnvironment() ? '' : 'packages/flutter_ondevice_asr/'}assets/transcribers/whisper/tokenizer/$vocabFileName';
    debugPrint("Loading vocab: $vocabFileName (language: $language)");
    await WhisperTokenizer.instance.loadVocab(path: vocabPath);

    // Load control tokens from generation config to get token IDs
    await _loadControlTokens();

    OrtEnv.instance.init();

    // Load Super Encoder (combined preprocessor + encoder)
    debugPrint("Loading Super Encoder Model...");
    final superEncoderPath = '$modelDirectory/super_encoder.onnx';
    final superEncoderBytes = await _loadBytes(superEncoderPath);
    superEncoderSession = onnxConfig.createSession(superEncoderBytes);
    debugPrint("Super Encoder loaded.");

    // Load decoder and decoder with past model
    debugPrint("Loading Decoder Model...");
    final decoderPath = '$modelDirectory/decoder_model.onnx';
    final decoderBytes = await _loadBytes(decoderPath);
    decoderSession = onnxConfig.createSession(decoderBytes);

    final decoderWithPastPath = '$modelDirectory/decoder_with_past_model.onnx';
    final decoderWithPastBytes = await _loadBytes(decoderWithPastPath);
    decoderWithPastSession = onnxConfig.createSession(decoderWithPastBytes);
    debugPrint("Decoder loaded.");

    debugPrint(">> All models loaded successfully! <<");
  }

  Future<void> _loadControlTokens() async {
    final configPath = '$modelDirectory/generation_config.json';
    final configFile = await _loadString(configPath);
    final config = jsonDecode(configFile) as Map<String, dynamic>;

    // Helper to extract int from either int or list format
    int getTokenId(dynamic value) {
      if (value is int) {
        return value;
      } else if (value is List && value.isNotEmpty) {
        return value[0] as int;
      }
      throw FormatException('Invalid token ID format: $value');
    }

    // Get basic token IDs (handle both int and list formats)
    sotToken = getTokenId(config['decoder_start_token_id']);
    eotToken = getTokenId(config['eos_token_id']);
    noTimestampsToken = getTokenId(config['no_timestamps_token_id']);

    // Parse forced_decoder_ids to get initial token sequence
    final forcedDecoderIds = config['forced_decoder_ids'] as List<dynamic>?;
    if (forcedDecoderIds != null && forcedDecoderIds.isNotEmpty) {
      for (var pair in forcedDecoderIds) {
        final position = pair[0] as int;
        final tokenId = pair[1] as int?;

        if (tokenId != null) {
          // Position 1 is typically language (for multilingual) or notimestamps (for en-only)
          // Position 2 is transcribe (for multilingual)
          if (position == 1 && tokenId != noTimestampsToken) {
            languageToken = tokenId;
          } else if (position == 2) {
            transcribeToken = tokenId;
          }
        }
      }
    }

    // If we have task_to_id, use it for transcribe token
    final taskToId = config['task_to_id'] as Map<String, dynamic>?;
    if (taskToId != null && taskToId.containsKey('transcribe')) {
      transcribeToken = taskToId['transcribe'] as int;
    }

    // If language is specified but we don't have languageToken from config, try lang_to_id
    if (language != null && languageToken == null) {
      final langToId = config['lang_to_id'] as Map<String, dynamic>?;
      if (langToId != null) {
        final langKey = '<|$language|>';
        languageToken = langToId[langKey] as int?;

        // Safety check: if language token still not found, throw a clear error
        if (languageToken == null) {
          throw StateError(
            'Language "$language" not found in model configuration. '
            'This model may not support this language. '
            'Available languages: ${langToId.keys.map((k) => k.replaceAll(RegExp(r'[<|>]'), '')).join(", ")}'
          );
        }
      }
    }

    debugPrint("Loaded token config: sot=$sotToken, eot=$eotToken, lang=$languageToken, transcribe=$transcribeToken, notimestamps=$noTimestampsToken");
  }


  List<OrtValue?> _runSuperEncoder(Float32List audio) {
    // Run super encoder (preprocessor + encoder combined)
    // Takes raw audio, outputs hidden states
    final runOptions = OrtRunOptions();
    final audioTensor = OrtValueTensor.createTensorWithDataList(audio, [1, audio.length]);
    final lengthTensor = OrtValueTensor.createTensorWithDataList(Int64List.fromList([audio.length]), [1]);

    final outputs = superEncoderSession!.run(
      runOptions,
      {
        'waveforms': audioTensor,
        'waveforms_lens': lengthTensor,
      },
    );

    audioTensor.release();
    lengthTensor.release();
    runOptions.release();

    return outputs;
  }


  Map<String, dynamic> _runDecoder(OrtValueTensor audioFeaturesTensor, int maxTokens, {bool computeConfidence = false}) {

    // // Build initial tokens list from config-loaded token IDs
    final List<int> tokens = [
      sotToken,
      languageToken!,
      transcribeToken!,
      noTimestampsToken];

    if (verbose) {
      debugPrint("Initial token IDs: sot=$sotToken, lang=$languageToken, transcribe=$transcribeToken, notimestamps=$noTimestampsToken");
    }

    Map<String, OrtValue>? pastKeyValues;
    Map<String, OrtValue>? encoderPastKeyValues; // Stays constant after first iteration
    final decoderOutputNames = decoderSession!.outputNames;
    final decoderWithPastOutputNames = decoderWithPastSession!.outputNames;
    final List<double> confidences = [];

    // Reusable buffers to avoid allocations in the loop
    final runOptions = OrtRunOptions();
    final singleTokenBuffer = Int64List(1);
    final decoderWithPastInputs = <String, OrtValue>{}; // Reusable map

    // Account for initial tokens (4) in max_length calculation
    // max_new_tokens = max_length - decoder_input_ids.shape[1]
    final maxNewTokens = maxTokens - tokens.length;

    for (int i = 0; i < maxNewTokens; ++i) {
      List<OrtValue?> decoderOutputs;
      List<double> lastLogits;

      if (pastKeyValues == null) {
        // First iteration - use full decoder with all tokens
        final tokensTensor = OrtValueTensor.createTensorWithDataList(
          Int64List.fromList(tokens),
          [1, tokens.length],
        );

        final decoderInputs = {
          'input_ids': tokensTensor,
          'encoder_hidden_states': audioFeaturesTensor,
        };

        decoderOutputs = decoderSession!.run(
          runOptions,
          decoderInputs,
        );

        tokensTensor.release();

        // Get logits - we compute for all 4 initial tokens but only use the last one
        // This is necessary to establish the KV cache for subsequent iterations
        final logits = decoderOutputs[0]?.value as List;
        lastLogits = (logits[0][tokens.length - 1] as List).cast<double>();

        // Release logits tensor after extracting values
        final logitsTensor = decoderOutputs[0];
        if (logitsTensor is OrtValueTensor) {
          logitsTensor.release();
        }

        pastKeyValues = {};
        encoderPastKeyValues = {};
        for (int j = 1; j < decoderOutputs.length; j++) {
          final outputName = decoderOutputNames[j];
          if (outputName.startsWith('present.')) {
            final pastName = outputName.replaceFirst('present.', 'past_key_values.');
            final ortValue = decoderOutputs[j]!;
            pastKeyValues[pastName] = ortValue;

            // Also store encoder past_key_values separately (they stay constant across the sequence)
            // later, decoderOutputs won't contain them hence we grab them once here
            if (outputName.contains('.encoder.')) {
              encoderPastKeyValues[pastName] = ortValue;
            }
          }
        }

        if (verbose) {
          debugPrint("First iteration: stored ${pastKeyValues.length} past_key_values total, with ${encoderPastKeyValues.length} from encoder.");
          debugPrint("First iteration logits shape: [${logits.length}, ${(logits[0] as List).length}, ${((logits[0] as List)[0] as List).length}]");
        }
      } else {
        // Store past key values from "present.*" outputs
        // Subsequent iterations - use decoder_with_past with only last token

        // Reuse buffer to avoid allocation
        singleTokenBuffer[0] = tokens.last;
        final currentTokenTensor = OrtValueTensor.createTensorWithDataList(
          singleTokenBuffer,
          [1, 1],
        );

        // Reuse map, clear and repopulate
        decoderWithPastInputs.clear();
        decoderWithPastInputs['input_ids'] = currentTokenTensor;
        decoderWithPastInputs.addAll(pastKeyValues);

        decoderOutputs = decoderWithPastSession!.run(
          runOptions,
          decoderWithPastInputs,
        );
        currentTokenTensor.release();

        // Get logits
        final logits = decoderOutputs[0]?.value as List;
        lastLogits = (logits[0][0] as List).cast<double>();

        // Release logits tensor after extracting values
        final logitsTensor = decoderOutputs[0];
        if (logitsTensor is OrtValueTensor) {
          logitsTensor.release();
        }

        // Update decoder past key values in place (encoder values stay constant)

        // Collect decoder keys to update
        final decoderKeys = <String>[];
        for (var key in pastKeyValues.keys) {
          if (key.contains('.decoder.')) {
            decoderKeys.add(key);
          }
        }

        // Release and remove old decoder past_key_values
        for (var key in decoderKeys) {
          final value = pastKeyValues[key];
          if (value is OrtValueTensor) {
            value.release();
          }
          pastKeyValues.remove(key);
        }

        // Add new decoder past_key_values from outputs (encoder values remain untouched)
        for (int j = 1; j < decoderOutputs.length; j++) {
          final outputName = decoderWithPastOutputNames[j];
          if (outputName.startsWith('present.') && outputName.contains('.decoder.')) {
            final pastName = outputName.replaceFirst('present.', 'past_key_values.');
            pastKeyValues[pastName] = decoderOutputs[j]!;
          }
        }
      }

      // Find token with highest probability
      // basically manual argmax here...
      int nextToken = 0;
      double maxLogit = lastLogits[0];
      for (int j = 1; j < lastLogits.length; j++) {
        if (lastLogits[j] > maxLogit) {
          maxLogit = lastLogits[j];
          nextToken = j;
        }
      }

      // Calculate confidence using softmax (only if requested)
      if (computeConfidence) {
        double expSum = 0.0;
        for (int j = 0; j < lastLogits.length; j++) {
          expSum += exp(lastLogits[j] - maxLogit);
        }
        final confidence = exp(lastLogits[nextToken] - maxLogit) / expSum;
        confidences.add(confidence);
      }

      if (verbose) {
        debugPrint("Step $i | ID: $nextToken | Logit: ${maxLogit.toStringAsFixed(2)} | Text: ${WhisperTokenizer.instance.decode(tokens, skipSpecialTokens: false)}");
      }

      // we stop either at eotToken or once we his the 32 token max
      if (nextToken == eotToken) {
        if (verbose) {
          debugPrint("--- Reached eotToken..");
        }
        break;
      }

      tokens.add(nextToken);
    }

    // Clean up reusable resources
    runOptions.release();

    // Clean up final past_key_values (always runs, whether we broke early or hit maxTokens)
    if (pastKeyValues != null) {
      for (var value in pastKeyValues.values) {
        if (value is OrtValueTensor) {
          value.release();
        }
      }
    }

    final transcript = WhisperTokenizer.instance.decode(tokens, skipSpecialTokens: true).trim();
    if (verbose) {
      debugPrint("Transcript: $transcript");
      if (computeConfidence) {
        debugPrint("Average confidence: ${confidences.isEmpty ? 0 : confidences.reduce((a, b) => a + b) / confidences.length}");
      }
    }

    return {
      'transcript': transcript,
      'confidences': confidences,
    };
  }

  @override
  Future<TranscriptionResult> transcribeFile(
    String path, {
    bool withConfidence = false,
    int? maxOutputTokens,
  }) async {
    final audio = await compute(Audio.instance.loadAudio, path);
    return transcribe(audio, withConfidence: withConfidence, maxOutputTokens: maxOutputTokens);
  }

  /// Transcribe audio to text
  ///
  /// Returns a [TranscriptionResult] with the transcribed text and metadata.
  /// If [withConfidence] is true, includes word-level confidence scores.
  @override
  Future<TranscriptionResult> transcribe(
    Float32List audio, {
    bool withConfidence = false,
    int? maxOutputTokens,
  }) async {
    final t1 = DateTime.now();

    // Run super encoder (preprocessor + encoder combined)
    final encoded = _runSuperEncoder(audio);
    final t2 = DateTime.now();

    // audio feature embeddings [1, seqLen, featureDim] are input needed for Whisper decoder
    // Super encoder output: [0] = features_lens, [1] = last_hidden_state
    final audioFeaturesTensor = encoded[1] as OrtValueTensor;
    final t3 = DateTime.now();

    // Calculate effective max tokens:
    // 1. Use explicitly provided value if given
    // 2. Otherwise calculate dynamically based on audio duration
    final audioDuration = audio.length / 16000.0;  // Assuming 16kHz sample rate
    final effectiveMaxTokens = maxOutputTokens ??
        (audioDuration * tokensPerSecond).ceil().clamp(10, 224);

    if (verbose) {
      debugPrint('>>> Audio: ${audioDuration.toStringAsFixed(2)}s → maxTokens: $effectiveMaxTokens ($tokensPerSecond tokens/sec)');
    }

    // Decode with or without confidence calculation
    final decoderResult = _runDecoder(audioFeaturesTensor, effectiveMaxTokens, computeConfidence: withConfidence);
    audioFeaturesTensor.release();
    final t4 = DateTime.now();

    if (verbose) {
      debugPrint(">>> TIME TO RUN SUPER ENCODER  : ${t2.difference(t1).inMilliseconds}ms");
      debugPrint(">>> TIME TO ENCODER CONVERSION : ${t3.difference(t2).inMilliseconds}ms");
      debugPrint(">>> TIME TO RUN DECODER        : ${t4.difference(t3).inMilliseconds}ms");
    }

    // Extract transcript and confidences from decoder result
    String transcript = decoderResult['transcript'] as String;
    List<WordConfidence>? wordConfidences;

    // Filter out BLANK_AUDIO token (always suppress for Whisper)
    if (transcript == '[BLANK_AUDIO]') {
      debugPrint('>>> BLANK_AUDIO detected - returning empty transcript');
      transcript = '';
    }

    // Build word confidences if available
    if (withConfidence && transcript.isNotEmpty) {
      final confidences = decoderResult['confidences'] as List<double>?;
      if (confidences != null) {
        final words = transcript.split(' ');
        // Match words with confidences (they should be same length)
        wordConfidences = List.generate(
          min(words.length, confidences.length),
          (i) => WordConfidence(word: words[i], confidence: confidences[i]),
        );
      }
    }

    final duration = audio.length / 16000.0; // Assuming 16kHz sample rate

    return TranscriptionResult(
      text: transcript,
      isFinal: true, // Non-streaming is always final
      duration: duration,
      timestamp: DateTime.now(),
      wordConfidences: wordConfidences,
    );
  }

  @Deprecated('Use transcribe(audio, withConfidence: true) instead')
  Future<TranscriptionResult> transcribeWithConfidence(Float32List audio) async {
    return transcribe(audio, withConfidence: true);
  }

  @override
  void dispose() {
    superEncoderSession?.release();
    decoderSession?.release();
    decoderWithPastSession?.release();
  }

  @override
  String get modelName => 'whisper-${language ?? "en"}';

  /// Get all supported languages for Whisper multilingual models.
  /// Use this to validate language codes before creating a WhisperTranscriber instance.
  static List<String> getAllSupportedLanguages() {
    return [
      'af', 'am', 'ar', 'as', 'az', 'ba', 'be', 'bg', 'bn', 'bo', 'br', 'bs',
      'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi',
      'fo', 'fr', 'gl', 'gu', 'ha', 'haw', 'he', 'hi', 'hr', 'ht', 'hu', 'hy',
      'id', 'is', 'it', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'la', 'lb',
      'ln', 'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt',
      'my', 'ne', 'nl', 'nn', 'no', 'oc', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru',
      'sa', 'sd', 'si', 'sk', 'sl', 'sn', 'so', 'sq', 'sr', 'su', 'sv', 'sw',
      'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'uk', 'ur', 'uz', 'vi',
      'yi', 'yo', 'zh',
    ];
  }
}
