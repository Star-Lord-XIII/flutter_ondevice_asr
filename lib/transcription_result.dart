/// Word-level transcription with confidence score and timing information
class Word {
  /// The transcribed word
  final String word;

  /// Confidence score for this word (0.0 to 1.0)
  final double confidence;

  /// Start time in seconds within the audio
  final double start;

  /// End time in seconds within the audio
  final double end;

  Word({
    required this.word,
    required this.confidence,
    required this.start,
    required this.end,
  });

  /// Convert to dictionary for JSON serialization
  Map<String, dynamic> toDict() {
    return {
      'word': word,
      'confidence': confidence,
      'start': start,
      'end': end,
    };
  }

  @override
  String toString() => '$word (${confidence.toStringAsFixed(3)}) [${start.toStringAsFixed(2)}s-${end.toStringAsFixed(2)}s]';
}

/// Unified transcription result (streaming and non-streaming)
class TranscriptionResult {
  /// The transcribed text
  final String text;

  /// Whether this is a final transcription (true) or partial (false)
  /// For non-streaming transcription, this is always true
  final bool isFinal;

  /// Duration of the audio segment in seconds
  final double duration;

  /// Timestamp when this result was produced
  final DateTime timestamp;

  /// Optional word-level transcription with timing and confidence
  final List<Word>? words;

  /// Text segments from ASR
  final List<String>? segments;

  /// Segment-level confidence scores
  final List<double>? confidences;

  /// Compression ratios per segment (if available)
  final List<double>? compressionRatios;

  TranscriptionResult({
    required this.text,
    required this.isFinal,
    required this.duration,
    required this.timestamp,
    this.words,
    this.segments,
    this.confidences,
    this.compressionRatios,
  });

  /// Average confidence across all words
  /// Returns null if words is not available
  double? get avgConfidence {
    if (words == null || words!.isEmpty) {
      return null;
    }
    final sum = words!.map((w) => w.confidence).reduce((a, b) => a + b);
    return sum / words!.length;
  }

  /// Average confidence across all segments
  /// Returns null if confidences is not available
  double? get avgSegmentConfidence {
    if (confidences == null || confidences!.isEmpty) {
      return null;
    }
    final sum = confidences!.reduce((a, b) => a + b);
    return sum / confidences!.length;
  }

  /// Convert to dictionary for JSON serialization
  Map<String, dynamic> toDict() {
    return {
      'text': text,
      'is_final': isFinal,
      'duration': duration,
      'timestamp': timestamp.toIso8601String(),
      'words': words?.map((w) => w.toDict()).toList(),
      'segments': segments,
      'confidences': confidences,
      'compression_ratios': compressionRatios,
    };
  }

  @override
  String toString() {
    final confStr = avgConfidence != null ? ', avgConf: ${avgConfidence!.toStringAsFixed(3)}' : '';
    return 'TranscriptionResult(text: "$text", isFinal: $isFinal, duration: ${duration.toStringAsFixed(2)}s$confStr)';
  }
}
