// Model-agnostic streaming transcriber implementation based on VAD segmentation.
// Should work with any Transcriber implementation.

import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter_ondevice_asr/onnx_config.dart';
import 'package:flutter_ondevice_asr/transcriber.dart';
import 'package:flutter_ondevice_asr/transcription_result.dart';
import 'package:flutter_ondevice_asr/vad/silero_vad.dart';

/// Model-agnostic streaming transcriber using VAD-based segmentation.
class StreamingTranscriber {
  final Transcriber transcriber;

  // Immutable VAD configuration (set at creation, and requires re-initialzing streaming instance to change)
  final int sampleRate;
  final double vadThreshold;
  final int eosMinSilence;
  final OnnxConfig? vadOnnxConfig;
  final bool verbose;

  // Mutable session parameters (set when starting recording)
  bool _enablePartials = true;
  int _minPartialDuration = 500;
  int _maxSegmentDuration = 10000;

  late final SileroVAD sileroModel;
  late final SileroVADIterator _sileroVad;
  late final StreamController<TranscriptionResult> _transcriptionController;

  final List<double> _speechBuffer = [];
  final List<double> _vadChunkBuffer = []; // Buffer for accumulating into 512-sample chunks for Silero VAD
  bool _isRecording = false;
  int _lastPartialBufferLength = 0; // Track buffer length at last partial transcription
  int _chunkCounter = 0;

  bool _isDisposed = false;

  // Private constructor
  StreamingTranscriber._({
    required this.transcriber,
    required this.sampleRate,
    required this.vadThreshold,
    required this.eosMinSilence,
    required this.vadOnnxConfig,
    required this.verbose,
  }) {
    _transcriptionController = StreamController<TranscriptionResult>.broadcast();
  }

  /// Create and initialize a new StreamingTranscriber instance
  ///
  /// Required:
  /// - [transcriber]: Any initialized Transcriber implementation (eg Whisper)
  ///
  /// Optional VAD parameters:
  /// - [vadThreshold]: VAD sensitivity, 0.0-1.0 (default: **0.5**)
  ///   - Higher = less sensitive (fewer false positives, may miss quiet speech)
  ///   - Lower = more sensitive (catches quiet speech, more false positives)
  /// - [eosMinSilence]: Silence duration in ms to end a segment (default: **300**)
  ///   - How long to wait after speech stops before finalizing. 
  ///   - Defaults are good for standard speech, but may need adjustment for particularly slow or fast speech.
  /// - [sampleRate]: Audio sample rate in Hz (default: **16000**)
  ///   - Must match your audio input
  ///
  /// Optional session parameters (can be changed later via `configure()`):
  /// - [enablePartials]: Emit partial transcriptions during speech (default: **true**)
  /// - [minPartialDuration]: Minimum ms between partial updates (default: **500**)
  /// - [maxSegmentDuration]: Maximum segment length in ms before forcing end (default: **10000**)
  /// - [vadOnnxConfig]: ONNX config for VAD model.
  /// - [verbose]: Enable verbose logging for debugging (default: **false**)
  static Future<StreamingTranscriber> create({
    required Transcriber transcriber,
    double vadThreshold = 0.5,
    int eosMinSilence = 300,
    int sampleRate = 16000,
    bool enablePartials = true,
    int minPartialDuration = 500,
    int maxSegmentDuration = 10000,
    OnnxConfig? vadOnnxConfig,
    bool verbose = false,
  }) async {
    final instance = StreamingTranscriber._(
      transcriber: transcriber,
      sampleRate: sampleRate,
      vadThreshold: vadThreshold,
      eosMinSilence: eosMinSilence,
      vadOnnxConfig: vadOnnxConfig,
      verbose: verbose,
    );

    // Set initial mutable parameters
    instance._enablePartials = enablePartials;
    instance._minPartialDuration = minPartialDuration;
    instance._maxSegmentDuration = maxSegmentDuration;

    await instance._initializeVAD();
    return instance;
  }

  Future<void> _initializeVAD() async {
    sileroModel = SileroVAD();

    // Use provided VAD config, or use VAD-specific lightweight default
    // Note: We DON'T default to transcriber config because VAD needs different settings
    final config = vadOnnxConfig ?? const VadOnnxConfig();
    await sileroModel.loadModel(config);

    _sileroVad = SileroVADIterator(
      model: sileroModel,
      threshold: vadThreshold,
      samplingRate: sampleRate,
      minSilenceDurationMs: eosMinSilence,
    );

    debugPrint('[Streaming] Using Silero VAD (threshold: $vadThreshold)');
  }

  /// Allow (re)-configuration of session parameters before processing audio
  void configure({
    bool? enablePartials,
    int? minPartialDuration,
    int? maxSegmentDuration,
  }) {
    if (enablePartials != null) _enablePartials = enablePartials;
    if (minPartialDuration != null) _minPartialDuration = minPartialDuration;
    if (maxSegmentDuration != null) _maxSegmentDuration = maxSegmentDuration;
  }

  /// Stream of transcription results (both partial and final)
  Stream<TranscriptionResult> get transcriptionStream => _transcriptionController.stream;

  /// Process an audio chunk and emit transcription results
  ///
  /// Audio must be Float32List in range [-1.0, 1.0] at the configured sample rate (default 16kHz)
  /// Note: For Silero VAD with 16kHz, audio will be buffered into 512-sample chunks automatically
  Future<void> processAudioChunk(Float32List audioChunk) async {
    if (_isDisposed) {
      throw StateError('StreamingTranscriber has been disposed');
    }

    // Add incoming audio to VAD buffer
    _vadChunkBuffer.addAll(audioChunk);

    // Process VAD in 512-sample chunks (matches Python logic exactly)
    while (_vadChunkBuffer.length >= sileroModel.requiredChunkSize) {
      // Extract exactly 512 samples for VAD
      final vadChunk = Float32List.fromList(_vadChunkBuffer.sublist(0, sileroModel.requiredChunkSize));
      _vadChunkBuffer.removeRange(0, sileroModel.requiredChunkSize);
      _chunkCounter++;

      // Add to speech buffer
      _speechBuffer.addAll(vadChunk);
      final bufferDuration = _speechBuffer.length / sampleRate;

      // Log periodically every 50 chunks (helps debug if audio is flowing)
      if (verbose && _chunkCounter % 50 == 0) {
        debugPrint('[Streaming] Chunk #$_chunkCounter: Buffer=${bufferDuration.toStringAsFixed(2)}s, Recording=$_isRecording');
      }

      // Process with VAD
      final vadEvent = _sileroVad.call(vadChunk);

      if (vadEvent != null) {
        if (vadEvent == 'start') {
          // Speech started - reset partial tracking
          _isRecording = true;
          _lastPartialBufferLength = 0;
          if (verbose) {
            debugPrint('[Streaming] Speech started (buffer: ${bufferDuration.toStringAsFixed(2)}s)');
          }
        } else if (vadEvent == 'end') {
          // End of segment detected by VAD
          _isRecording = false;
          if (verbose) {
            debugPrint('[Streaming] Speech ended (buffer: ${bufferDuration.toStringAsFixed(2)}s)');
          }

          if (_speechBuffer.isNotEmpty) {
            final audioData = Float32List.fromList(_speechBuffer);
            final result = await transcriber.transcribe(audioData, withConfidence: false);

            // Only emit if we got actual text
            if (result.text.isNotEmpty) {
              // Create streaming result (override duration from transcribe to use actual buffer duration)
              final streamingResult = TranscriptionResult(
                text: result.text,
                isFinal: true,
                duration: bufferDuration,
                timestamp: DateTime.now(),
                wordConfidences: result.wordConfidences,
              );

              if (!_transcriptionController.isClosed) {
                _transcriptionController.add(streamingResult);
              }

              if (verbose) {
                debugPrint('[Streaming] Final (${bufferDuration.toStringAsFixed(2)}s): ${result.text}');
              }
            }
          }

          // Reset buffer AFTER transcription finishes
          _speechBuffer.clear();

          // Break out of loop - let next processAudioChunk() call handle accumulated chunks
          break;
        }
      } else {
        // No VAD event - check if we should send partial or force end
        if (_isRecording) {
          // Calculate how much NEW audio has accumulated since last partial
          final newAudioSamples = _speechBuffer.length - _lastPartialBufferLength;
          final newAudioDuration = newAudioSamples / sampleRate;

          // Force end if segment is too long
          if (bufferDuration * 1000 > _maxSegmentDuration) {
            if (verbose) {
              debugPrint('[Streaming] Max segment duration reached, forcing end');
            }

            final audioData = Float32List.fromList(_speechBuffer);
            final result = await transcriber.transcribe(audioData, withConfidence: false);

            // Only emit if we got actual text
            if (result.text.isNotEmpty) {
              final streamingResult = TranscriptionResult(
                text: result.text,
                isFinal: true,
                duration: bufferDuration,
                timestamp: DateTime.now(),
                wordConfidences: result.wordConfidences,
              );

              if (!_transcriptionController.isClosed) {
                _transcriptionController.add(streamingResult);
              }

              if (verbose) {
                debugPrint('[Streaming] Final (forced, ${bufferDuration.toStringAsFixed(2)}s): ${result.text}');
              }
            }

            // Clear buffer completely (treat like VAD end event)
            _speechBuffer.clear();
            _lastPartialBufferLength = 0;

            // Break out of loop - let next processAudioChunk() call handle accumulated chunks
            break;
          }
          // Send partial transcription if enough NEW audio has accumulated since last partial
          else if (_enablePartials && newAudioDuration * 1000 > _minPartialDuration) {
            if (verbose) {
              debugPrint('[Streaming] Sending partial transcription');
            }

            final audioData = Float32List.fromList(_speechBuffer);
            final result = await transcriber.transcribe(audioData, withConfidence: false);

            // Only emit if we got actual text
            if (result.text.isNotEmpty) {
              final streamingResult = TranscriptionResult(
                text: result.text,
                isFinal: false,
                duration: bufferDuration,
                timestamp: DateTime.now(),
                wordConfidences: result.wordConfidences,
              );

              if (!_transcriptionController.isClosed) {
                _transcriptionController.add(streamingResult);
              }

              if (verbose) {
                debugPrint('[Streaming] Partial (${bufferDuration.toStringAsFixed(2)}s): ${result.text}');
              }
            }

            // Update tracking - we've now transcribed up to current buffer length
            // Keep the buffer growing for next partial/final
            _lastPartialBufferLength = _speechBuffer.length;

            // Break out of loop - let next processAudioChunk() call handle accumulated chunks
            break;
          }
        } else {
          // Not recording - keep only small buffer to avoid cutting off speech start
          final emptyFramesToKeep = (0.1 * sampleRate).toInt();
          if (_speechBuffer.length > emptyFramesToKeep) {
            _speechBuffer.removeRange(0, _speechBuffer.length - emptyFramesToKeep);
          }
        }
      }
    }
  }

  /// Flush any remaining audio in the buffer (call this when stopping the stream)
  Future<void> flush() async {
    if (_speechBuffer.isNotEmpty) {
      final bufferDuration = _speechBuffer.length / sampleRate;
      final audioData = Float32List.fromList(_speechBuffer);

      final result = await transcriber.transcribe(audioData, withConfidence: false);

      // Only emit if we got actual text
      if (result.text.isNotEmpty) {
        final streamingResult = TranscriptionResult(
          text: result.text,
          isFinal: true,
          duration: bufferDuration,
          timestamp: DateTime.now(),
          wordConfidences: result.wordConfidences,
        );

        if (!_transcriptionController.isClosed) {
          _transcriptionController.add(streamingResult);
        }

        if (verbose) {
          debugPrint('[Streaming] Flush final (${bufferDuration.toStringAsFixed(2)}s): ${result.text}');
        }
      }
      _speechBuffer.clear();
    }
  }

  /// Reset the streaming state (clears buffers and VAD state)
  void reset() {
    _speechBuffer.clear();
    _vadChunkBuffer.clear();
    _isRecording = false;
    _lastPartialBufferLength = 0;
    _chunkCounter = 0;

    _sileroVad.resetStates();

    if (verbose) {
      debugPrint('[Streaming] State reset');
    }
  }

  /// Dispose of resources
  Future<void> dispose() async {
    if (_isDisposed) return;

    _isDisposed = true;

    // Flush any remaining audio
    await flush();

    await _transcriptionController.close();

    // Dispose VAD resources
    _sileroVad.model.dispose();

    if (verbose) {
      debugPrint('[Streaming] Disposed');
    }
  }
}
