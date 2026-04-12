import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';
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

    final vocabFile = File(path);
    // if (!vocabFile.existsSync()) {
    //   _logger.warning('Vocab file <$path> does not exist.');
    //   return Result.error(Exception('vocab file <$path> does not exist.'));
    // }
    final vocabJson = await _loadString(vocabFile.path);
    final vocab = jsonDecode(vocabJson) as Map<String, dynamic>;
    _idToToken = vocab.map((key, value) => MapEntry(value as int, key));
    _initByteDecoder();
    return Result.ok(null);
  }

  /// Load string from either external file or bundled asset
  Future<String> _loadString(String path) async {
    if (Utils.isExternalPath(path)) {
      return await File(path).readAsString();
    } else {
      final assetPath = '${Utils.isRunningInTestEnvironment() ? '' : 'packages/flutter_ondevice_asr/'}$path';
      return await rootBundle.loadString(assetPath);
    }
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
}
