/// Dart implementation of Silero VAD using ONNX Runtime
/// Based on https://github.com/snakers4/silero-vad

import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_ondevice_asr/onnx_config.dart';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';

/// ONNX wrapper for Silero VAD model
class SileroVAD {
  final String sileroVadModelPath = 'packages/flutter_ondevice_asr/assets/vad/silero_vad/silero_vad.onnx'; 

  late final OrtSession _session;
  late final List<String> _inputNames;
  late final List<String> _outputNames;

  // //final int requiredChunkSize = config.sampleRate == 16000 ? 512 : 256;
  final int requiredChunkSize = 512;

  final int contextSize = 64;

  Float32List? _h;
  Float32List? _c;
  Float32List? _state;
  Float32List _context = Float32List(0);
  int _lastSr = 0;
  int _lastBatchSize = 0;

  bool _isInitialized = false;

  /// Create default ONNX configuration for VAD
  /// VAD is lightweight, so we use minimal threading
  static OnnxConfig createDefaultConfig() {
    return const VadOnnxConfig();
  }

  /// Load Silero VAD model from asset path
  ///
  /// [onnxConfig] - Optional ONNX configuration. If not provided, uses lightweight defaults optimized for VAD.
  Future<void> loadModel([OnnxConfig? onnxConfig]) async {
    final config = onnxConfig ?? createDefaultConfig();

    final rawAssetFile = await rootBundle.load(sileroVadModelPath);
    final bytes = rawAssetFile.buffer.asUint8List();
    _session = config.createSession(bytes);

    _inputNames = _session.inputNames;
    _outputNames = _session.outputNames;

    resetStates();
    _isInitialized = true;

    if (kDebugMode) {
      debugPrint('[SileroVAD] Model loaded from: $sileroVadModelPath');
      debugPrint('[SileroVAD] Inputs: $_inputNames');
      debugPrint('[SileroVAD] Outputs: $_outputNames');
    }
  }

  /// Reset internal model states
  void resetStates({int batchSize = 1}) {
    // Support both state formats (separate h/c or combined state)
    _h = Float32List(2 * batchSize * 64);
    _c = Float32List(2 * batchSize * 64);
    _state = Float32List(2 * batchSize * 128);
    _context = Float32List(0);
    _lastSr = 0;
    _lastBatchSize = 0;
  }

  /// Run VAD inference on audio chunk
  ///
  /// Args:
  ///   x: Audio chunk as Float32List (normalized to [-1, 1])
  ///   sr: Sample rate (8000 or 16000)
  ///
  /// Returns:
  ///   Speech probability p in [0.0, 1.0]
  double call(Float32List x, int sr) {
    if (!_isInitialized) {
      throw StateError('Model not loaded. Call loadModel() first.');
    }

    // Validate sample rate
    if (sr != 16000 && sr != 8000) {
      throw ArgumentError('Unsupported sampling rate: $sr. Supported: 8000, 16000');
    }

    // Expected chunk sizes
    if (x.length != requiredChunkSize) {
      throw ArgumentError('Expected $requiredChunkSize samples for ${sr}Hz, got ${x.length}');
    }

    final batchSize = 1;

    // Reset states if batch size or sample rate changed
    if (_lastBatchSize == 0) {
      resetStates(batchSize: batchSize);
    }
    if (_lastSr != 0 && _lastSr != sr) {
      resetStates(batchSize: batchSize);
    }
    if (_lastBatchSize != 0 && _lastBatchSize != batchSize) {
      resetStates(batchSize: batchSize);
    }

    // Initialize context buffer if needed
    if (_context.isEmpty) {
      _context = Float32List(contextSize);
    }

    // Concatenate context with input: shape (1, contextSize + requiredChunkSize)
    final inputLength = contextSize + requiredChunkSize;
    final inputData = Float32List(inputLength);
    for (int i = 0; i < contextSize; i++) {
      inputData[i] = _context[i];
    }
    for (int i = 0; i < requiredChunkSize; i++) {
      inputData[contextSize + i] = x[i];
    }

    // Create 2D tensor: (batch_size=1, samples) - pass Float32List directly
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      inputData,  // Pass Float32List directly without wrapping
      [1, inputLength],
    );

    final srTensor = OrtValueTensor.createTensorWithDataList(
      Int64List.fromList([sr]),
      [1],
    );

    try {
      Map<String, OrtValue> inputs;
      List<OrtValue?> outputs;

      // Build inputs based on model's expected input names
      if (_inputNames.contains('state')) {
        // Combined state version - use Float32List directly
        final stateTensor = OrtValueTensor.createTensorWithDataList(
          _state!,
          [2, 1, 128],
        );

        try {
          inputs = {
            'input': inputTensor,
            'state': stateTensor,
            'sr': srTensor,
          };

          final runOptions = OrtRunOptions();
          try {
            outputs = _session.run(
              runOptions,
              inputs,
            );
          } finally {
            runOptions.release();
          }

          // Update state from outputs
          if (outputs.length >= 2 && outputs[1] != null) {
            final stateValue = outputs[1]!.value;
            if (stateValue is Float32List) {
              _state = stateValue;
            } else if (stateValue is List) {
              // Fallback: flatten nested list structure
              final flatState = Float32List(2 * 128);
              int idx = 0;
              final stateList = stateValue as List<List<List<double>>>;
              for (var i = 0; i < 2; i++) {
                for (var j = 0; j < 128; j++) {
                  flatState[idx++] = stateList[i][0][j];
                }
              }
              _state = flatState;
            }
          }
        } finally {
          // Release state tensor
          stateTensor.release();
        }
      } else {
        // Separate h/c version - use Float32List directly
        final hTensor = OrtValueTensor.createTensorWithDataList(
          _h!,
          [2, 1, 64],
        );

        final cTensor = OrtValueTensor.createTensorWithDataList(
          _c!,
          [2, 1, 64],
        );

        try {
          inputs = {
            'input': inputTensor,
            'h': hTensor,
            'c': cTensor,
            'sr': srTensor,
          };

          final runOptions = OrtRunOptions();
          try {
            outputs = _session.run(
              runOptions,
              inputs,
            );
          } finally {
            runOptions.release();
          }

          // Update h and c from outputs
          if (outputs.length >= 3) {
            if (outputs[1] != null) {
              final hValue = outputs[1]!.value;
              if (hValue is Float32List) {
                _h = hValue;
              } else if (hValue is List) {
                // Fallback: flatten nested list structure
                final hList = hValue as List<List<List<double>>>;
                final flatH = Float32List(2 * 64);
                int idx = 0;
                for (var i = 0; i < 2; i++) {
                  for (var j = 0; j < 64; j++) {
                    flatH[idx++] = hList[i][0][j];
                  }
                }
                _h = flatH;
              }
            }
            if (outputs[2] != null) {
              final cValue = outputs[2]!.value;
              if (cValue is Float32List) {
                _c = cValue;
              } else if (cValue is List) {
                // Fallback: flatten nested list structure
                final cList = cValue as List<List<List<double>>>;
                final flatC = Float32List(2 * 64);
                int idx = 0;
                for (var i = 0; i < 2; i++) {
                  for (var j = 0; j < 64; j++) {
                    flatC[idx++] = cList[i][0][j];
                  }
                }
                _c = flatC;
              }
            }
          }
        } finally {
          // Release h/c tensors
          hTensor.release();
          cTensor.release();
        }
      }

      // Get output probability
      double speechProb = 0.0;
      if (outputs.isNotEmpty && outputs[0] != null) {
        final outValue = outputs[0]!.value;
        if (outValue is List<List<double>>) {
          speechProb = outValue[0][0];
        } else if (outValue is List<double>) {
          speechProb = outValue[0];
        } else if (outValue is double) {
          speechProb = outValue;
        }
      }

      // Update context with last contextSize samples from input
      _context = Float32List(contextSize);
      _context.setRange(0, contextSize, inputData, inputData.length - contextSize);

      _lastSr = sr;
      _lastBatchSize = batchSize;

      // Release output tensors
      for (var output in outputs) {
        output?.release();
      }

      return speechProb;
    } finally {
      // Always release input and sr tensors
      inputTensor.release();
      srTensor.release();
    }
  }

  /// Release resources
  void dispose() {
    if (_isInitialized) {
      _session.release();
      _isInitialized = false;
    }
  }
}

/// VAD iterator for streaming speech detection
class SileroVADIterator {
  final SileroVAD model;
  final double threshold;
  final int samplingRate;
  final int minSilenceDurationMs;
  final int speechPadMs;

  late final int _minSilenceSamples;

  bool _triggered = false;
  int _tempEnd = 0;
  int _currentSample = 0;
  int _callCount = 0;

  SileroVADIterator({
    required this.model,
    this.threshold = 0.5,
    this.samplingRate = 16000,
    this.minSilenceDurationMs = 100,
    this.speechPadMs = 30,
  }) {
    _minSilenceSamples = samplingRate * minSilenceDurationMs ~/ 1000;
  }

  /// Reset all iterator states
  void resetStates({bool fullReset = true}) {
    if (fullReset) {
      model.resetStates();
    }
    _triggered = false;
    _tempEnd = 0;
    _currentSample = 0;
  }

  /// Process audio chunk and detect speech boundaries
  ///
  /// Args:
  ///   x: Audio chunk as Float32List (normalized to [-1, 1])
  ///
  /// Returns:
  ///   'start' when speech begins, 'end' when speech ends, null otherwise
  String? call(Float32List x) {
    final windowSizeSamples = x.length;

    // Get speech probability from model
    final speechProb = model.call(x, samplingRate);

    // Debug: Log probability periodically
    _callCount++;
    if (kDebugMode && _callCount % 20 == 0) {
      debugPrint('[SileroVAD] Prob: ${speechProb.toStringAsFixed(3)}, Triggered: $_triggered, TempEnd: $_tempEnd');
    }

    if (speechProb >= threshold && _tempEnd != 0) {
      // Reset tempEnd if speech resumes during silence
      _tempEnd = 0;
    }

    if (speechProb >= threshold && !_triggered) {
      // Speech start detected
      _triggered = true;
      _currentSample += windowSizeSamples;

      if (kDebugMode) {
        debugPrint('[SileroVAD] Speech started (prob: ${speechProb.toStringAsFixed(3)})');
      }

      return 'start';
    }

    if (speechProb < threshold - 0.15 && _triggered) {
      // Potential silence detected
      if (_tempEnd == 0) {
        // Mark the start of silence
        _tempEnd = _currentSample;
      }

      // Check if silence has lasted long enough
      if (_currentSample - _tempEnd >= _minSilenceSamples) {
        // Speech end confirmed
        _tempEnd = 0;
        _triggered = false;
        _currentSample += windowSizeSamples;

        if (kDebugMode) {
          debugPrint('[SileroVAD] Speech ended (prob: ${speechProb.toStringAsFixed(3)})');
        }

        return 'end';
      }
    }

    _currentSample += windowSizeSamples;
    return null;
  }
}
