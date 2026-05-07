import 'package:flutter/material.dart';

import '../models/ward.dart';
import 'risk_gauge.dart';

/// Compact card showing ward name + heat/flood risk gauges.
/// Tapping navigates to WardDetailScreen.
class WardCard extends StatelessWidget {
  final WardRisk ward;
  final VoidCallback? onTap;

  const WardCard({super.key, required this.ward, this.onTap});

  Color get _borderColor {
    switch (ward.riskLevel) {
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

  @override
  Widget build(BuildContext context) {
    return Card(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: _borderColor, width: 1.5),
      ),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              // Ward name + risk badge
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Expanded(
                    child: Text(
                      ward.displayName,
                      style: const TextStyle(
                        fontWeight: FontWeight.w700,
                        fontSize: 14,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  _RiskBadge(level: ward.riskLevel),
                ],
              ),
              const SizedBox(height: 8),
              // Gauges row
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  RiskGauge(
                    value: ward.heatStressScore,
                    label: 'Heat',
                    size: 70,
                  ),
                  RiskGauge(
                    value: ward.floodRiskScore,
                    label: 'Flood',
                    size: 70,
                  ),
                ],
              ),
              // LST chip
              if (ward.lstPredCelsius != null)
                Padding(
                  padding: const EdgeInsets.only(top: 4),
                  child: Row(
                    children: [
                      const Icon(Icons.thermostat, size: 14,
                          color: Colors.deepOrange),
                      const SizedBox(width: 4),
                      Text(
                        'LST ${ward.lstPredCelsius!.toStringAsFixed(1)}°C',
                        style: const TextStyle(fontSize: 12),
                      ),
                      if (ward.ndvi != null) ...[
                        const SizedBox(width: 12),
                        const Icon(Icons.park, size: 14, color: Colors.green),
                        const SizedBox(width: 4),
                        Text(
                          'NDVI ${ward.ndvi!.toStringAsFixed(2)}',
                          style: const TextStyle(fontSize: 12),
                        ),
                      ],
                    ],
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _RiskBadge extends StatelessWidget {
  final String level;
  const _RiskBadge({required this.level});

  Color get _color {
    switch (level) {
      case 'critical': return const Color(0xFFD32F2F);
      case 'high':     return const Color(0xFFF57C00);
      case 'medium':   return const Color(0xFFFBC02D);
      default:         return const Color(0xFF388E3C);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: _color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: _color),
      ),
      child: Text(
        level.toUpperCase(),
        style: TextStyle(
          color: _color,
          fontSize: 10,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.5,
        ),
      ),
    );
  }
}
