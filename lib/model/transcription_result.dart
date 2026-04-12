import 'word.dart';

class TranscriptionResult {
  final String text;
  final bool isFinal;
  final double durationInSeconds;
  final DateTime timestamp;
  final List<Word>? words;
  final List<String>? segments;
  final List<double>? confidences;
  final List<double>? compressionRatios;

  const TranscriptionResult({
    required this.text,
    required this.isFinal,
    required this.durationInSeconds,
    required this.timestamp,
    this.words,
    this.segments,
    this.confidences,
    this.compressionRatios,
  });
}
