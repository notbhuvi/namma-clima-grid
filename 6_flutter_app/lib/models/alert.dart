import 'package:flutter/material.dart';

/// Real-time ward alert pushed over WebSocket.
class WardAlert {
  final int wardId;
  final String alertType;
  final String severity;
  final String message;
  final double value;
  final double threshold;
  final DateTime timestamp;
  final String? source;    // "bbmp_official" | "citizen_image" | null
  final String? imageUrl;  // for citizen image flood reports

  const WardAlert({
    required this.wardId,
    required this.alertType,
    required this.severity,
    required this.message,
    required this.value,
    required this.threshold,
    required this.timestamp,
    this.source,
    this.imageUrl,
  });

  factory WardAlert.fromJson(Map<String, dynamic> j) => WardAlert(
        wardId:    j['ward_id'] as int,
        alertType: j['alert_type'] as String,
        severity:  j['severity'] as String,
        message:   j['message'] as String,
        value:     (j['value'] as num).toDouble(),
        threshold: (j['threshold'] as num).toDouble(),
        timestamp: DateTime.parse(j['timestamp'] as String),
        source:    j['source'] as String?,
        imageUrl:  j['image_url'] as String?,
      );

  Color get severityColor {
    switch (severity) {
      case 'critical':
        return const Color(0xFFD32F2F);
      case 'high':
        return const Color(0xFFF57C00);
      case 'medium':
        return const Color(0xFFFBC02D);
      default:
        return const Color(0xFF388E3C);
    }
  }

  IconData get alertIcon {
    switch (alertType) {
      case 'heat_wave':
        return Icons.thermostat;
      case 'flood_warning':
        return Icons.water;
      case 'air_quality':
        return Icons.air;
      default:
        return Icons.warning_amber;
    }
  }
}
