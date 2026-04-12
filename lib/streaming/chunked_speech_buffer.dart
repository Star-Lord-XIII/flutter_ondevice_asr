import 'dart:collection';

import 'package:flutter/foundation.dart';

class ChunkedSpeechBuffer {
  final Queue<Float32List> _chunks = Queue();
  int _totalSamples = 0;

  int get length => _totalSamples;

  void addChunk(Float32List chunk) {
    _chunks.addLast(chunk);
    _totalSamples += chunk.length;
  }

  void clear() {
    _chunks.clear();
    _totalSamples = 0;
  }

  void removeFromFront(int samples) {
    while ((_totalSamples - _chunks.first.length) >= samples) {
      _totalSamples -= _chunks.first.length;
      _chunks.removeFirst();
    }
  }

  /// Flatten to Float32List for transcription
  Float32List toFloat32List() {
    final out = Float32List(_totalSamples);
    int offset = 0;
    for (final chunk in _chunks) {
      out.setAll(offset, chunk);
      offset += chunk.length;
    }
    return out;
  }
}
