import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:onnxruntime_v2/onnxruntime_v2.dart';

/// TODO: might want to use different execution providers:
/// VAD may benefit from dedicated NPU execution providers (NNAPI on Android, CoreML on iOS) for ultra-low latency inference.

/// Configuration for ONNX Runtime sessions.
class OnnxConfig {
  /// Number of threads to use for intra-op parallelism.
  ///
  /// Default: 2 threads for better compatibility with low-end devices.
  /// Higher thread counts can cause contention on dual-core devices.
  final int intraOpNumThreads;

  /// Number of threads to use for inter-op parallelism.
  ///
  /// Default: 1 thread.
  /// Only increase this if you have multiple independent operations to parallelize.
  final int interOpNumThreads;

  /// Graph optimization level.
  ///
  /// Default: [GraphOptimizationLevel.ortEnableAll] - enables all optimizations.
  /// Can be set to [GraphOptimizationLevel.ortEnableBasic] for faster model loading
  /// at the cost of slower inference.
  final GraphOptimizationLevel graphOptimizationLevel;

  /// Enable XNNPACK execution provider for ARM devices.
  ///
  /// Default: true - automatically enables on Android/iOS for ARM-optimized SIMD math.
  /// XNNPACK provides significant performance improvements on mobile devices.
  /// Falls back gracefully to CPU provider if unavailable.
  final bool enableXnnpack;

  /// Create ONNX Runtime configuration.
  const OnnxConfig({
    this.intraOpNumThreads = 2,
    this.interOpNumThreads = 1,
    this.graphOptimizationLevel = GraphOptimizationLevel.ortEnableAll,
    this.enableXnnpack = true,
  });

  /// Create ONNX Runtime session options from this configuration.
  /// Make sure to release after use!
  OrtSessionOptions createSessionOptions() {
    final options = OrtSessionOptions()
      ..setIntraOpNumThreads(intraOpNumThreads)
      ..setInterOpNumThreads(interOpNumThreads)
      ..setSessionGraphOptimizationLevel(graphOptimizationLevel);

    // Add XNNPACK execution provider if enabled and on ARM device
    if (enableXnnpack && (Platform.isAndroid || Platform.isIOS)) {
      try {
        options.appendXnnpackProvider();
        debugPrint(
          '[OnnxConfig] Execution Provider: XNNPACK ✓ (ARM-optimized SIMD)',
        );
      } catch (e) {
        debugPrint(
          '[OnnxConfig] Execution Provider: CPU (XNNPACK unavailable: $e)',
        );
      }
    } else {
      debugPrint(
        '[OnnxConfig] Execution Provider: CPU (platform: ${Platform.operatingSystem})',
      );
    }

    debugPrint(
      '[OnnxConfig] Threads: intra=$intraOpNumThreads, inter=$interOpNumThreads',
    );
    debugPrint('[OnnxConfig] Optimization: $graphOptimizationLevel');

    return options;
  }

  /// Create an ONNX Runtime session from model bytes with automatic options cleanup.
  ///
  /// This helper method ensures that session options are properly released after
  /// session creation, preventing native memory leaks.
  ///
  /// [modelBytes]: The ONNX model file contents as bytes.
  /// Returns: A configured [OrtSession] ready for inference.
  OrtSession createSession(Uint8List modelBytes) {
    final options = createSessionOptions();
    try {
      return OrtSession.fromBuffer(modelBytes, options);
    } finally {
      options.release();
    }
  }

  @override
  String toString() {
    return 'OnnxConfig(threads: $intraOpNumThreads/$interOpNumThreads, '
        'optimization: $graphOptimizationLevel, xnnpack: $enableXnnpack)';
  }
}

/// Specialized ONNX configuration for ASR transcriber models.
///
/// Transcriber models are compute-intensive and benefit from multi-threading.
class TranscriberOnnxConfig extends OnnxConfig {
  const TranscriberOnnxConfig({
    super.intraOpNumThreads = 2,
    super.interOpNumThreads = 1,
    super.graphOptimizationLevel = GraphOptimizationLevel.ortEnableAll,
    super.enableXnnpack = true,
  });
}

class SuperEncoderConfig extends OnnxConfig {
  const SuperEncoderConfig({
    super.intraOpNumThreads = 8,
    super.interOpNumThreads = 1,
    super.graphOptimizationLevel = GraphOptimizationLevel.ortEnableAll,
    super.enableXnnpack = true,
  });
}

/// Specialized ONNX configuration for VAD (Voice Activity Detection) models.
///
/// VAD models are lightweight and fast, so they use minimal resources.
class VadOnnxConfig extends OnnxConfig {
  const VadOnnxConfig({
    super.intraOpNumThreads = 1,
    super.interOpNumThreads = 1,
    super.graphOptimizationLevel = GraphOptimizationLevel.ortEnableAll,
    super.enableXnnpack = true,
  });
}
