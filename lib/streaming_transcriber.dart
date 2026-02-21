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
  bool _isRecordingSpeech = false;
  int _chunkCounter = 0;
  double _bufferDurationSeconds = 0.0; // Total duration in buffer since VAD start
  double _newAudioDurationSeconds = 0.0; // Duration of new audio since last partial transcription
  late final double _vadChunkDurationSeconds; // Duration of one VAD chunk, calculated once
  bool _transcriptionInProgress = false; // Track if transcription is currently running
  bool _bufferContainsSpeech = false; // Track if buffer contains untranscribed speech (not just silence padding)

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
  ///   This will trigger a transcriber call whenever enough data for a partial is collected 
  ///   (len >= minPartialDuration) and especially for short minPartialDuration this will lead
  ///   to significant system use. For weaker devices, it will be important to set minPartialDuration
  ///   conservatively (ie, high). However, in order for transcriptions to feel real-time we would
  ///   ideally set minPartialDuration to 300ms.
  /// - [minPartialDuration]: Minimum ms between partial updates (default: **500**)
  /// - [maxSegmentDuration]: Maximum segment length in ms before forcing end (default: **30000**)
  ///   We limit this to the maximum segment length, Whisper can natively handle. We intentionally
  ///   skip any sort of sliding window approaches in the streaming-based transcription for efficiency.
  /// - [vadOnnxConfig]: ONNX config for VAD model.
  /// - [verbose]: Enable verbose logging for debugging (default: **false**)
  static Future<StreamingTranscriber> create({
    required Transcriber transcriber,
    double vadThreshold = 0.5,
    int eosMinSilence = 300,
    int sampleRate = 16000,
    bool enablePartials = true,
    int minPartialDuration = 500,
    int maxSegmentDuration = 30000,
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
    sileroModel = SileroVAD(verbose: verbose);

    // Use provided VAD config, or use VAD-specific lightweight default
    // Note: We DON'T default to transcriber config because VAD needs different settings
    final config = vadOnnxConfig ?? const VadOnnxConfig();
    await sileroModel.loadModel(config);

    _sileroVad = SileroVADIterator(
      model: sileroModel,
      threshold: vadThreshold,
      samplingRate: sampleRate,
      minSilenceDurationMs: eosMinSilence,
      verbose: verbose,
    );

    // Calculate chunk duration once (same for all chunks)
    _vadChunkDurationSeconds = sileroModel.requiredChunkSize / sampleRate;

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

    // Process VAD in 512-sample (sileroModel.requiredChunkSize) chunks
    while (_vadChunkBuffer.length >= sileroModel.requiredChunkSize) {
      final vadChunk = Float32List.fromList(_vadChunkBuffer.sublist(0, sileroModel.requiredChunkSize));
      _vadChunkBuffer.removeRange(0, sileroModel.requiredChunkSize);
      _chunkCounter++;

      // Add to speech buffer and update duration counters
      _speechBuffer.addAll(vadChunk);
      _bufferDurationSeconds += _vadChunkDurationSeconds;
      _newAudioDurationSeconds += _vadChunkDurationSeconds;
      final bufferDuration = _bufferDurationSeconds;

      // Log periodically every 50 chunks (helps debug if audio is flowing)
      if (verbose && _chunkCounter % 50 == 0) {
        debugPrint('[Streaming] Chunk #$_chunkCounter: Buffer=${bufferDuration.toStringAsFixed(2)}s, RecordingSpeech=$_isRecordingSpeech');
      }

      // Process with VAD
      final vadEvent = _sileroVad.call(vadChunk);

      if (vadEvent != null) {
        if (vadEvent == 'start') {
          // Speech started - mark as recording speech
          _isRecordingSpeech = true;
          _bufferContainsSpeech = true; // Buffer now contains speech
          // Only reset partial tracking if no transcription is in progress
          // If transcription is running, we want to keep track of accumulated untranscribed audio
          if (!_transcriptionInProgress) {
            _newAudioDurationSeconds = 0.0;
          }
          if (verbose) {
            debugPrint('[Streaming] Speech started (buffer: ${bufferDuration.toStringAsFixed(2)}s)');
          }
        } else if (vadEvent == 'end') {
          // End of segment detected by VAD - should transcribe
          _isRecordingSpeech = false;
          if (verbose) {
            debugPrint('[Streaming] Speech ended (buffer: ${bufferDuration.toStringAsFixed(2)}s)');
          }

          if (_speechBuffer.isNotEmpty && !_transcriptionInProgress) {
            // Copy buffer for async transcription
            final audioToTranscribe = Float32List.fromList(_speechBuffer);
            final duration = bufferDuration;

            // Clear buffer immediately - VAD end means segment is complete
            _speechBuffer.clear();
            _bufferDurationSeconds = 0.0;
            _newAudioDurationSeconds = 0.0;
            _bufferContainsSpeech = false; // Buffer cleared, no more untranscribed speech

            // Run transcription asynchronously without blocking
            _transcriptionInProgress = true;
            _transcribeAsync(audioToTranscribe, duration, isFinal: true).then((_) {
              _transcriptionInProgress = false;
            });
          }
          // If transcription is in progress, keep accumulating buffer - don't discard speech
        }
      } else {
        // No "new" VAD event - check if we have collected enough speech data for a partial to transcribe
        // or whether we need to force end because we're hitting the max partial length
        if (_isRecordingSpeech) {
          // Force end if segment is too long
          if (bufferDuration * 1000 >= _maxSegmentDuration) {
            if (verbose) {
              debugPrint('[Streaming] Max segment duration reached, forcing end');
            }

            if (!_transcriptionInProgress) {
              // Copy buffer for async transcription
              final audioToTranscribe = Float32List.fromList(_speechBuffer);
              final duration = bufferDuration;

              // Clear buffer - max segment forces final transcription
              _speechBuffer.clear();
              _bufferDurationSeconds = 0.0;
              _newAudioDurationSeconds = 0.0;
              _bufferContainsSpeech = false; // Buffer cleared, no more untranscribed speech

              // Run transcription asynchronously without blocking
              _transcriptionInProgress = true;
              _transcribeAsync(audioToTranscribe, duration, isFinal: true).then((_) {
                _transcriptionInProgress = false;
              });
            }
            // If transcription is in progress, keep accumulating buffer - don't discard speech
          }
          // Send partial transcription if enough NEW audio has accumulated since last partial
          // only if _enablePartials !
          else if (_enablePartials && _newAudioDurationSeconds * 1000 >= _minPartialDuration) {
            if (!_transcriptionInProgress) {
              if (verbose) {
                debugPrint('[Streaming] Sending partial transcription');
              }

              // Copy ENTIRE buffer for cumulative partial transcription
              final audioToTranscribe = Float32List.fromList(_speechBuffer);
              final duration = bufferDuration;

              // DO NOT clear _speechBuffer - partials are cumulative within a VAD segment!
              // Only reset new audio counter
              // Note: _bufferContainsSpeech stays true since buffer is not cleared for partials
              _newAudioDurationSeconds = 0.0;

              // Run transcription asynchronously without blocking
              _transcriptionInProgress = true;
              _transcribeAsync(audioToTranscribe, duration, isFinal: false).then((_) {
                _transcriptionInProgress = false;
              });
            }
            // else: transcription in progress, keep accumulating and will trigger again after it completes
          }
          else {
            if (verbose) {
              debugPrint('[Streaming] Collecting speech (buffer: ${bufferDuration.toStringAsFixed(2)}s)');
            }
          }
        } else {
          // Not recording speech - but check if we have stale buffered speech that needs transcribing
          if (_bufferContainsSpeech && !_transcriptionInProgress &&
              bufferDuration * 1000 >= _maxSegmentDuration) {
            if (verbose) {
              debugPrint('[Streaming] Stale buffered speech exceeded max segment duration, forcing transcription');
            }

            // Transcribe the stale buffered speech
            final audioToTranscribe = Float32List.fromList(_speechBuffer);
            final duration = bufferDuration;

            _speechBuffer.clear();
            _bufferDurationSeconds = 0.0;
            _newAudioDurationSeconds = 0.0;
            _bufferContainsSpeech = false;

            _transcriptionInProgress = true;
            _transcribeAsync(audioToTranscribe, duration, isFinal: true).then((_) {
              _transcriptionInProgress = false;
            });
          }

          // Keep only small buffer of empty frames (no speech as detected by VAD)
          // to avoid cutting off speech start
          // BUT: Don't trim if we have untranscribed speech waiting in the buffer
          if (!_bufferContainsSpeech) {
            final emptyFramesToKeep = (0.1 * sampleRate).toInt();
            if (_speechBuffer.length > emptyFramesToKeep) {
              _speechBuffer.removeRange(0, _speechBuffer.length - emptyFramesToKeep);
            }
          }
        }
      }
    }
  }

  /// Flush any remaining audio in the buffer (call this when stopping the stream)
  Future<void> flush() async {
    // Wait for any in-progress transcription to complete
    while (_transcriptionInProgress) {
      await Future.delayed(const Duration(milliseconds: 10));
    }

    if (_speechBuffer.isNotEmpty) {
      final audioData = Float32List.fromList(_speechBuffer);

      final result = await transcriber.transcribe(
        audioData,
        segmentEnd: true,
        getWordDetails: false,
      );

      // Only emit if we got actual text
      if (result.text.isNotEmpty) {
        if (!_transcriptionController.isClosed) {
          _transcriptionController.add(result);
        }

        if (verbose) {
          debugPrint('[Streaming] Flush final (${result.duration.toStringAsFixed(2)}s): ${result.text}');
        }
      }
      _speechBuffer.clear();
      _bufferDurationSeconds = 0.0;
      _bufferContainsSpeech = false;
    }
  }

  /// Helper method to run transcription asynchronously without blocking audio processing
  Future<void> _transcribeAsync(Float32List audio, double duration, {required bool isFinal}) async {
    try {
      final result = await transcriber.transcribe(
        audio,
        segmentEnd: isFinal,
        getWordDetails: false,
      );

      // Only emit if we got actual text
      if (result.text.isNotEmpty) {
        if (!_transcriptionController.isClosed) {
          _transcriptionController.add(result);
        }

        if (verbose) {
          final label = result.isFinal ? 'Final' : 'Partial';
          debugPrint('[Streaming] $label (${result.duration.toStringAsFixed(2)}s): ${result.text}');
        }
      }
    } catch (e) {
      if (verbose) {
        debugPrint('[Streaming] Transcription error: $e');
      }
    }
  }

  /// Reset the streaming state (clears buffers and VAD state)
  void reset() {
    _speechBuffer.clear();
    _vadChunkBuffer.clear();
    _bufferDurationSeconds = 0.0;
    _newAudioDurationSeconds = 0.0;
    _isRecordingSpeech = false;
    _bufferContainsSpeech = false;
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
