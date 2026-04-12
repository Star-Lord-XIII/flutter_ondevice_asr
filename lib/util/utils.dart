import 'dart:io';

class Utils {

  /// adjust paths for tests vs regular use
  static bool isRunningInTestEnvironment() {
    return Platform.environment.containsKey('FLUTTER_TEST');
  }


  /// Helper to determine if path is external (absolute file path) vs bundled asset
  static bool isExternalPath(String path) {
    return path.startsWith('/') || path.startsWith('file://');
  }
}
