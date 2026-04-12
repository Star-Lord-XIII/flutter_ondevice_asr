/// Word-level transcription with confidence score and timing information.
class Word {
  final String word;
  final double confidence;
  final double start;
  final double end;

  const Word({
    required this.word,
    required this.confidence,
    required this.start,
    required this.end,
  });
}
