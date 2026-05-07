import 'dart:async';
import 'dart:convert';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import '../models/alert.dart';
import 'base_url.dart';

final wsServiceProvider = Provider<WebSocketService>((ref) => WebSocketService());

class WebSocketService {
  WebSocketChannel? _channel;

  // Accumulated alert list — emits the full list on every new message
  final List<WardAlert> _alerts = [];
  final _controller = StreamController<List<WardAlert>>.broadcast();

  bool _connected = false;
  Timer? _pingTimer;
  Timer? _reconnectTimer;

  Stream<List<WardAlert>> get alertStream => _controller.stream;
  bool get isConnected => _connected;

  void connect() {
    if (_connected) return;
    _reconnectTimer?.cancel();

    final url = '$webSocketBaseUrl/ws/alerts';
    try {
      _channel = WebSocketChannel.connect(Uri.parse(url));
      _connected = true;

      _channel!.stream.listen(
        _onMessage,
        onError: _onError,
        onDone: _onDone,
        cancelOnError: false,
      );

      // Keepalive ping every 15 seconds
      _pingTimer?.cancel();
      _pingTimer = Timer.periodic(const Duration(seconds: 15), (_) {
        _send({'ping': true});
      });
    } catch (_) {
      _connected = false;
      _scheduleReconnect();
    }
  }

  void disconnect() {
    _pingTimer?.cancel();
    _reconnectTimer?.cancel();
    _channel?.sink.close();
    _connected = false;
  }

  void _send(Map<String, dynamic> data) {
    try {
      _channel?.sink.add(jsonEncode(data));
    } catch (_) {}
  }

  void _onMessage(dynamic raw) {
    try {
      final data = jsonDecode(raw as String) as Map<String, dynamic>;
      if (data.containsKey('pong')) return;

      final incoming = (data['alerts'] as List? ?? [])
          .map((e) => WardAlert.fromJson(e as Map<String, dynamic>))
          .toList();

      if (incoming.isEmpty) return;

      // Prepend new alerts, deduplicate, keep last 50
      final existingKeys = _alerts
          .map((a) => '${a.wardId}_${a.timestamp.millisecondsSinceEpoch}')
          .toSet();

      for (final alert in incoming.reversed) {
        final key = '${alert.wardId}_${alert.timestamp.millisecondsSinceEpoch}';
        if (!existingKeys.contains(key)) {
          _alerts.insert(0, alert);
          existingKeys.add(key);
        }
      }

      // Keep max 50
      if (_alerts.length > 50) _alerts.removeRange(50, _alerts.length);

      if (!_controller.isClosed) {
        _controller.add(List.unmodifiable(_alerts));
      }
    } catch (_) {}
  }

  void _onError(Object _) {
    _connected = false;
    _pingTimer?.cancel();
    _scheduleReconnect();
  }

  void _onDone() {
    _connected = false;
    _pingTimer?.cancel();
    _scheduleReconnect();
  }

  void _scheduleReconnect() {
    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(const Duration(seconds: 4), connect);
  }

  void dispose() {
    _pingTimer?.cancel();
    _reconnectTimer?.cancel();
    _channel?.sink.close();
    _controller.close();
  }
}
