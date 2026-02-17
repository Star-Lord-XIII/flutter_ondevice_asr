// Main library exports for flutter_ondevice_asr
// This package provides on-device ASR transcription with a model-agnostic architecture

// Core abstractions
export 'transcriber.dart';
export 'transcription_result.dart';
export 'streaming_transcriber.dart';

// ONNX Runtime configuration
export 'onnx_config.dart';

// Model implementations
export 'models/whisper/whisper_transcriber.dart';

// Shared utilities
export 'audio.dart';
export 'vad/silero_vad.dart';
