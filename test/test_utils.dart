import 'dart:io';

/// Convert relative asset path to absolute filesystem path for unit tests.
/// This ensures tests use the filesystem directly, not the asset bundle.
String toAbsolutePath(String relativePath) {
  return '${Directory.current.path}/$relativePath';
}
