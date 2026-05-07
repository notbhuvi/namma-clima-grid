import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../models/alert.dart';

class AlertBanner extends StatelessWidget {
  final WardAlert alert;
  final VoidCallback? onDismiss;

  const AlertBanner({super.key, required this.alert, this.onDismiss});

  bool get _isBBMP    => alert.source == 'bbmp_official';
  bool get _isCitizen => alert.source == 'citizen_image';

  @override
  Widget build(BuildContext context) {
    return Dismissible(
      key: ValueKey('${alert.wardId}_${alert.timestamp.millisecondsSinceEpoch}'),
      direction: DismissDirection.endToStart,
      onDismissed: (_) => onDismiss?.call(),
      background: Container(
        color: Colors.red.shade100,
        alignment: Alignment.centerRight,
        padding: const EdgeInsets.only(right: 16),
        child: const Icon(Icons.delete_outline, color: Colors.red),
      ),
      child: Card(
        margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
          side: BorderSide(
            color: _isBBMP
                ? const Color(0xFF1B6B3A).withOpacity(0.6)
                : alert.severityColor.withOpacity(0.4),
            width: _isBBMP ? 1.5 : 1,
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Source badge strip
            if (_isBBMP || _isCitizen)
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
                decoration: BoxDecoration(
                  color: _isBBMP
                      ? const Color(0xFF1B6B3A).withOpacity(0.1)
                      : Colors.blue.shade50,
                  borderRadius: const BorderRadius.vertical(top: Radius.circular(12)),
                ),
                child: Row(children: [
                  Icon(
                    _isBBMP ? Icons.verified_user : Icons.camera_alt,
                    size: 12,
                    color: _isBBMP ? const Color(0xFF1B6B3A) : Colors.blue.shade700,
                  ),
                  const SizedBox(width: 5),
                  Text(
                    _isBBMP ? 'Official BBMP Alert' : 'Citizen Photo Report — AI Detected',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      color: _isBBMP ? const Color(0xFF1B6B3A) : Colors.blue.shade700,
                    ),
                  ),
                ]),
              ),

            // Main tile
            ListTile(
              leading: CircleAvatar(
                backgroundColor: alert.severityColor.withOpacity(0.15),
                child: Icon(alert.alertIcon, color: alert.severityColor, size: 20),
              ),
              title: Text(
                alert.message,
                style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
              ),
              subtitle: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    DateFormat('HH:mm · dd MMM').format(alert.timestamp.toLocal()),
                    style: const TextStyle(fontSize: 11),
                  ),
                  if (alert.value > 0)
                    Text(
                      'Ward ${alert.wardId} · Score: ${alert.value.toStringAsFixed(1)}',
                      style: TextStyle(fontSize: 11, color: Colors.grey.shade500),
                    ),
                ],
              ),
              isThreeLine: alert.value > 0,
              trailing: Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: alert.severityColor.withOpacity(0.12),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  alert.severity.toUpperCase(),
                  style: TextStyle(
                    color: alert.severityColor,
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ),

            // Citizen image thumbnail (if present)
            if (_isCitizen && alert.imageUrl != null)
              Padding(
                padding: const EdgeInsets.fromLTRB(12, 0, 12, 10),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: Image.network(
                    '${_baseUrl()}${alert.imageUrl}',
                    height: 120,
                    width: double.infinity,
                    fit: BoxFit.cover,
                    errorBuilder: (_, __, ___) => const SizedBox.shrink(),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  String _baseUrl() {
    try {
      // ignore: avoid_web_libraries_in_flutter
      final h = Uri.base.origin;
      return h;
    } catch (_) {
      return 'http://localhost:8000';
    }
  }
}
