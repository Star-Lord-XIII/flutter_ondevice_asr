import 'dart:collection';
import 'dart:typed_data';

class Float32RingBuffer {
  final Queue<double> _queue = Queue();

  int get length => _queue.length;

  void addAll(Float32List data) {
    for (final sample in data) {
      _queue.addLast(sample);
    }
  }

  /// Drain exactly [count] samples into a new Float32List.
  /// Caller must check length >= count first.
  Float32List consume(int count) {
    final out = Float32List(count);
    for (int i = 0; i < count; i++) {
      out[i] = _queue.removeFirst();
    }
    return out;
  }

  void removeFromFront(int count) {
    for (int i = 0; i < count; i++) {
      _queue.removeFirst();
    }
  }

  void keepTail(int count) {
    while (_queue.length > count) {
      _queue.removeFirst();
    }
  }

  void clear() {
    _queue.clear();
  }
}
