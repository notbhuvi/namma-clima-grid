import 'package:flutter/foundation.dart';

const String _apiBaseUrlFromEnv = String.fromEnvironment(
  'API_BASE_URL',
  defaultValue: '',
);

String get apiBaseUrl {
  if (_apiBaseUrlFromEnv.isNotEmpty) {
    return _apiBaseUrlFromEnv;
  }

  if (kIsWeb) {
    return Uri.base.origin;
  }

  switch (defaultTargetPlatform) {
    case TargetPlatform.android:
      // Android emulators access the host machine through 10.0.2.2.
      return 'http://10.0.2.2:8000';
    case TargetPlatform.iOS:
    case TargetPlatform.macOS:
    case TargetPlatform.windows:
    case TargetPlatform.linux:
    case TargetPlatform.fuchsia:
      return 'http://localhost:8000';
  }
}

String get webSocketBaseUrl {
  final base = apiBaseUrl;
  if (base.startsWith('https://')) {
    return 'wss://${base.substring('https://'.length)}';
  }
  if (base.startsWith('http://')) {
    return 'ws://${base.substring('http://'.length)}';
  }
  return base;
}
