import 'dart:io';

import 'package:flutter/services.dart';

class Utils {
  /// Package name constant - update this if the package is renamed
  static const String packageName = 'flutter_ondevice_asr';

  /// Package asset prefix for loading bundled assets
  static const String _packageAssetPrefix = 'packages/$packageName/';

  /// adjust paths for tests vs regular use
  static bool isRunningInTestEnvironment() {
    return Platform.environment.containsKey('FLUTTER_TEST');
  }

  /// Helper to determine if path is a bundled asset.
  /// Asset paths start with 'assets/' - the 'packages/' prefix is added
  /// internally by getAssetPath(). Absolute filesystem paths (e.g. /Users/...)
  /// won't match this check and will use File() instead.
  static bool isAssetPath(String path) {
    return path.startsWith('assets/');
  }

  /// Get the asset path with package prefix (if not in test environment)
  static String getAssetPath(String path) {
    return '${isRunningInTestEnvironment() ? '' : _packageAssetPrefix}$path';
  }

  /// Load string from either bundled asset or external file
  static Future<String> loadString(String path) async {
    if (isAssetPath(path)) {
      return await rootBundle.loadString(getAssetPath(path));
    } else {
      return await File(path).readAsString();
    }
  }

  /// Load bytes from either bundled asset or external file
  static Future<Uint8List> loadBytes(String path) async {
    if (isAssetPath(path)) {
      final data = await rootBundle.load(getAssetPath(path));
      return data.buffer.asUint8List();
    } else {
      return await File(path).readAsBytes();
    }
  }
}
