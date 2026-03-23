// Tokenizer for whisper models (en-only and multilingual)
import 'dart:convert';
import 'package:flutter/services.dart';

class WhisperTokenizer {
  static WhisperTokenizer? _instance;
  Map<int, String>? _idToToken;
  Map<int, int>? _byteDecoder;

  WhisperTokenizer._();

  static WhisperTokenizer get instance {
    _instance ??= WhisperTokenizer._();
    return _instance!;
  }

  /// Load vocabulary from the given path
  Future<void> loadVocab({required String path}) async {
    if (_idToToken != null) return;

    final vocabJson = await rootBundle.loadString(path);
    final vocab = jsonDecode(vocabJson) as Map<String, dynamic>;

    _idToToken = {};
    for (var entry in vocab.entries) {
      _idToToken![entry.value as int] = entry.key;
    }

    _initByteDecoder();
  }

  /// Reset the tokenizer (for testing purposes)
  void reset() {
    _idToToken = null;
    _byteDecoder = null;
  }

  void _initByteDecoder() {
    _byteDecoder = {};
    final Map<int, int> b2u = {};

    // Helper to add ranges exactly like HuggingFace's bytes_to_unicode
    void addRange(int start, int end) {
      for (int i = start; i <= end; i++) {
        b2u[i] = i;
      }
    }

    addRange(33, 126);   // ! to ~
    addRange(161, 172);  // ¡ to ¬
    addRange(174, 255);  // ® to ÿ

    // Any byte not added above (including 0-32, 127-160, 173)
    // is mapped to Unicode characters starting at 256
    int n = 0;
    for (int b = 0; b < 256; b++) {
      if (!b2u.containsKey(b)) {
        b2u[b] = 256 + n;
        n++;
      }
    }

    // IMPORTANT: For decoding, we want: Unicode Character -> Raw Byte
    b2u.forEach((byte, unicode) {
      _byteDecoder![unicode] = byte;
    });
  }

  /// The correctly integrated decode method
  String decode(List<int> tokenIds, {bool skipSpecialTokens = true}) {
    if (_idToToken == null) {
      throw StateError('Tokenizer not loaded. Call loadVocab() first.');
    }

    final result = StringBuffer();
    final allBytes = <int>[];

    for (var id in tokenIds) {
      final token = _idToToken![id];
      if (token == null) continue;

      // Handle special tokens separately - they should be returned as-is
      if (token.startsWith('<|') && token.endsWith('|>')) {
        // Flush any accumulated bytes first
        if (allBytes.isNotEmpty) {
          result.write(utf8.decode(allBytes, allowMalformed: true));
          allBytes.clear();
        }

        // Add special token directly (or skip if requested)
        if (!skipSpecialTokens) {
          result.write(token);
        }
        continue;
      }

      // Convert characters in token string (e.g. 'Ġ', '你好') back to raw bytes
      for (int i = 0; i < token.length; i++) {
        final charUnit = token.codeUnitAt(i);
        final byte = _byteDecoder![charUnit];
        if (byte != null) {
          allBytes.add(byte);
        } else {
          // Fallback for characters that are already valid bytes
          if (charUnit < 256) allBytes.add(charUnit);
        }
      }
    }

    // Flush any remaining bytes
    if (allBytes.isNotEmpty) {
      result.write(utf8.decode(allBytes, allowMalformed: true));
    }

    return result.toString();
  }
}
