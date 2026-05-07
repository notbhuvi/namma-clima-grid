import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

/// Semicircular gauge displaying a 0–100 risk score.
class RiskGauge extends StatelessWidget {
  final double value;        // 0–100
  final String label;
  final double size;

  const RiskGauge({
    super.key,
    required this.value,
    required this.label,
    this.size = 120,
  });

  Color get _trackColor {
    if (value >= 75) return const Color(0xFFD32F2F);
    if (value >= 50) return const Color(0xFFF57C00);
    if (value >= 25) return const Color(0xFFFBC02D);
    return const Color(0xFF388E3C);
  }

  @override
  Widget build(BuildContext context) {
    final pct = value.clamp(0, 100) / 100;
    return SizedBox(
      width: size,
      height: size * 0.75,
      child: Stack(
        alignment: Alignment.center,
        children: [
          PieChart(
            PieChartData(
              startDegreeOffset: 180,
              sectionsSpace: 0,
              centerSpaceRadius: size * 0.30,
              sections: [
                PieChartSectionData(
                  value: pct,
                  color: _trackColor,
                  radius: size * 0.18,
                  showTitle: false,
                ),
                PieChartSectionData(
                  value: 1 - pct,
                  color: Colors.grey.shade200,
                  radius: size * 0.18,
                  showTitle: false,
                ),
              ],
            ),
          ),
          Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                value.toStringAsFixed(0),
                style: TextStyle(
                  fontSize: size * 0.22,
                  fontWeight: FontWeight.w800,
                  color: _trackColor,
                ),
              ),
              Text(
                label,
                style: TextStyle(
                  fontSize: size * 0.10,
                  color: Colors.grey.shade600,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
