import 'dart:typed_data';
import 'onnx_config.dart';
import 'transcription_result.dart';

/// Abstract base class for on-device speech recognition models.
abstract class Transcriber {
  /// Load the model files and initialize the transcriber.
  Future<void> loadModels();

  /// Transcribe audio data to text.
  ///
  /// Parameters:
  /// - [audio]: Audio samples as Float32List, normalized to [-1.0, 1.0] range
  /// - [withConfidence]: If true, include word-level confidence scores in result
  /// - [maxOutputTokens]: Optional maximum number of tokens to generate (null = auto)
  ///
  /// Returns a [TranscriptionResult] containing the transcribed text.
  /// For streaming transcription, use [StreamingTranscriber] instead.
  Future<TranscriptionResult> transcribe(
    Float32List audio, {
    bool withConfidence = false,
    int? maxOutputTokens,
  });

  /// Transcribe an audio file to text.
  ///
  /// Convenience method that loads the audio file and calls [transcribe].
  /// Supports WAV format with 16-bit PCM encoding.
  ///
  /// Parameters:
  /// - [path]: Path to the audio file
  /// - [withConfidence]: If true, include word-level confidence scores in result
  /// - [maxOutputTokens]: Optional maximum number of tokens to generate (null = auto)
  Future<TranscriptionResult> transcribeFile(
    String path, {
    bool withConfidence = false,
    int? maxOutputTokens,
  });

  /// Release all resources held by the transcriber.
  void dispose();

  /// Get the name of the underlying model.
  String get modelName;

  /// Get the ONNX Runtime configuration used by this transcriber.
  OnnxConfig get onnxConfig;
}
