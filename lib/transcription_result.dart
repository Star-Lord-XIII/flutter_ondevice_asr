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

  /// Optional word-level confidence scores
  /// Only available when transcription is performed with confidence enabled
  final List<WordConfidence>? wordConfidences;

  TranscriptionResult({
    required this.text,
    required this.isFinal,
    required this.duration,
    required this.timestamp,
    this.wordConfidences,
  });

  /// Average confidence across all words
  /// Returns null if wordConfidences is not available
  double? get avgConfidence {
    if (wordConfidences == null || wordConfidences!.isEmpty) {
      return null;
    }
    final sum = wordConfidences!.map((w) => w.confidence).reduce((a, b) => a + b);
    return sum / wordConfidences!.length;
  }

  @override
  String toString() {
    final confStr = avgConfidence != null ? ', avgConf: ${avgConfidence!.toStringAsFixed(3)}' : '';
    return 'TranscriptionResult(text: "$text", isFinal: $isFinal, duration: ${duration.toStringAsFixed(2)}s$confStr)';
  }
}

/// Word-level confidence score
class WordConfidence {
  /// The transcribed word
  final String word;

  /// Confidence score for this word (0.0 to 1.0)
  final double confidence;

  WordConfidence({
    required this.word,
    required this.confidence,
  });

  @override
  String toString() => '$word (${confidence.toStringAsFixed(3)})';
}
