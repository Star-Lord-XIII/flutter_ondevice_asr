import 'dart:io';

// adjust paths for tests vs regular use
bool isRunningInTestEnvironment() {
  return Platform.environment.containsKey('FLUTTER_TEST');
}