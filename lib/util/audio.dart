// Tooling for audio loading
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'dart:developer' as dev;

import 'utils.dart';

final class Audio {
  static Audio instance = Audio._init();

  Audio._init();

  /// Converts a WAV file to Float32List
  /// Returns normalized audio samples in range [-1.0, 1.0]
  Future<Float32List> loadAudio(String filePath) async {
    final tik = DateTime.now();
    // Read the WAV file as bytes
    final Uint8List bytes;
    if (Utils.isRunningInTestEnvironment()) {
      bytes = (await rootBundle.load(filePath)).buffer.asUint8List();
    } else {
      bytes = await File(filePath).readAsBytes();
    }
    // Parse WAV header (standard 44 bytes)
    final dataOffset = _findDataChunk(bytes);

    // Get audio data (skip header)
    final audioData = bytes.sublist(dataOffset);

    // Convert 16-bit PCM to Float32
    final output = _convert16BitToFloat32(audioData);
    final tok = DateTime.now();
    dev.log("AUDIO LOADED: ${tok.difference(tik).inMilliseconds}");
    return output;
  }

  /// Find the "data" chunk in WAV file
  int _findDataChunk(Uint8List bytes) {
    // Look for "data" chunk marker (0x64617461)
    for (int i = 12; i < bytes.length - 4; i++) {
      if (bytes[i] == 0x64 &&
          bytes[i + 1] == 0x61 &&
          bytes[i + 2] == 0x74 &&
          bytes[i + 3] == 0x61) {
        // Return position after "data" marker and size (8 bytes total)
        return i + 8;
      }
    }
    // If no "data" chunk found, assume standard 44-byte header
    return 44;
  }

  /// Convert 16-bit PCM to Float32 (most common format)
  Float32List _convert16BitToFloat32(Uint8List data) {
    final numSamples = data.length ~/ 2;
    final result = Float32List(numSamples);
    for (int i = 0; i < numSamples; i++) {
      // Read 16-bit signed integer (little-endian)
      final sample = (data[i * 2] | (data[i * 2 + 1] << 8));
      // Convert to signed
      final signed = sample > 32767 ? sample - 65536 : sample;
      // Normalize to [-1.0, 1.0]
      result[i] = signed / 32768.0;
    }
    return result;
  }
}
