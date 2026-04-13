import 'dart:convert';

import 'package:logging/logging.dart';

import '../../common/result.dart';
import '../../util/utils.dart';

class WhisperTokenizer {
  final _logger = Logger('WhisperTokenizer');

  Map<int, String> _idToToken = {};
  Map<int, int> _byteDecoder = {};

  Future<Result<void>> loadVocab({required String path}) async {
    _logger.fine('Loading vocab from path: <$path>');
    _idToToken = {};

    final vocabJson = await Utils.loadString(path);
    final vocab = jsonDecode(vocabJson) as Map<String, dynamic>;
    _idToToken = vocab.map((key, value) => MapEntry(value as int, key));
    _initByteDecoder();
    return Result.ok(null);
  }

  void _initByteDecoder() {
    _byteDecoder = {};
    final Map<int, int> byteToUnicode = {};

    void addRange(int start, int end) {
      for (int i = start; i <= end; i++) {
        byteToUnicode[i] = i;
      }
    }

    addRange("!".codeUnitAt(0), "~".codeUnitAt(0));
    addRange("¡".codeUnitAt(0), "¬".codeUnitAt(0));
    addRange("®".codeUnitAt(0), "ÿ".codeUnitAt(0));

    // Any byte not added above (including 0-32, 127-160, 173)
    // is mapped to Unicode characters starting at 256
    int n = 0;
    for (int b = 0; b < 256; b++) {
      if (!byteToUnicode.containsKey(b)) {
        byteToUnicode[b] = 256 + n;
        n++;
      }
    }

    // IMPORTANT: For decoding, we want: Unicode Character -> Raw Byte
    byteToUnicode.forEach((byte, unicode) {
      _byteDecoder[unicode] = byte;
    });
  }

  Result<String> decode(List<int> tokenIds, {bool skipSpecialTokens = true}) {
    if (_idToToken.isEmpty) {
      return Result.error(
        Exception('Tokenizer not loaded. Call loadVocab() first.'),
      );
    }
    final result = StringBuffer();
    final allBytes = <int>[];

    for (final id in tokenIds) {
      final token = _idToToken[id];
      if (token == null) {
        continue;
      } else if (_isSpecialToken(token)) {
        if (allBytes.isNotEmpty) {
          result.write(utf8.decode(allBytes, allowMalformed: true));
          allBytes.clear();
        }
        if (!skipSpecialTokens) {
          result.write(token);
        }
      } else {
        for (int i = 0; i < token.length; i++) {
          final charUnit = token.codeUnitAt(i);
          final byte = _byteDecoder[charUnit];
          if (byte != null) {
            allBytes.add(byte);
          } else {
            if (charUnit < 256) {
              allBytes.add(charUnit);
            }
          }
        }
      }
    }

    if (allBytes.isNotEmpty) {
      result.write(utf8.decode(allBytes, allowMalformed: true));
    }

    return Result.ok(result.toString());
  }

  bool _isSpecialToken(String token) {
    return token.startsWith('<|') && token.endsWith('|>');
  }

  /// Decode a single token to its text representation.
  /// Handles the BPE byte encoding used by Whisper.
  String decodeSingleToken(int tokenId, {bool skipSpecialTokens = true}) {
    final token = _idToToken[tokenId];
    if (token == null) return '';
    if (_isSpecialToken(token)) {
      return skipSpecialTokens ? '' : token;
    }

    final bytes = <int>[];
    for (int i = 0; i < token.length; i++) {
      final charUnit = token.codeUnitAt(i);
      final byte = _byteDecoder[charUnit];
      if (byte != null) {
        bytes.add(byte);
      } else if (charUnit < 256) {
        bytes.add(charUnit);
      }
    }

    return utf8.decode(bytes, allowMalformed: true);
  }

  /// Check if a token starts a new word (has the Ġ prefix indicating leading space).
  /// The Ġ character (Unicode 288) represents a space in Whisper's BPE encoding.
  bool tokenStartsNewWord(int tokenId) {
    final token = _idToToken[tokenId];
    if (token == null || _isSpecialToken(token)) return false;
    // Ġ is Unicode 288, which represents a leading space in BPE
    return token.isNotEmpty && token.codeUnitAt(0) == 288;
  }
}
