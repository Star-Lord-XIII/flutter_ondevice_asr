import 'package:flutter/foundation.dart';

import '../common/result.dart';
import 'model/transcription_result.dart';
import 'models/whisper/whisper_transcriber.dart';
import 'transcriber_type.dart';

abstract class Transcriber {
  static Transcriber getInstance(TranscriberType type) {
    switch (type) {
      case TranscriberType.whisper:
        return WhisperTranscriber();
    }
  }

  String? get modelPath;

  Future<Result<void>> loadModel({
    required String modelDirectory,
    required String languageCode,
    double tokensPerSecond,
  });

  Future<Result<TranscriptionResult>> transcribe(
    Float32List audio, {
    bool segmentEnd = true,
    bool getWordDetails = false,
    bool getSegmentDetails = false,
    int? maxOutputTokens,
  });

  Future<Result<TranscriptionResult>> transcribeFile(
    String path, {
    bool segmentEnd = true,
    bool getWordDetails = false,
    bool getSegmentDetails = false,
    int? maxOutputTokens,
  });

  void dispose();
}
