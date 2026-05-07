import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../providers/ward_providers.dart';
import '../widgets/risk_gauge.dart';

class WardDetailScreen extends ConsumerWidget {
  final int wardId;
  const WardDetailScreen({super.key, required this.wardId});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final wardAsync     = ref.watch(wardDetailProvider(wardId));
    final forecastAsync = ref.watch(wardForecastProvider(wardId));

    return Scaffold(
      appBar: AppBar(
        title: wardAsync.maybeWhen(
          data: (w) => Text(w.displayName),
          orElse: () => Text('Ward $wardId'),
        ),
      ),
      body: wardAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (e, _) => Center(child: Text('Error: $e')),
        data: (ward) => ListView(
          padding: const EdgeInsets.all(16),
          children: [
            // Risk gauges
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    const Text('Current Risk',
                        style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16)),
                    const SizedBox(height: 12),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        RiskGauge(value: ward.heatStressScore, label: 'Heat Stress', size: 130),
                        RiskGauge(value: ward.floodRiskScore,  label: 'Flood Risk',  size: 130),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 8),

            // Metadata chips
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Wrap(
                  spacing: 8,
                  runSpacing: 6,
                  children: [
                    if (ward.lstPredCelsius != null)
                      _MetaChip(
                          icon: Icons.thermostat,
                          label: 'LST ${ward.lstPredCelsius!.toStringAsFixed(1)}°C',
                          color: Colors.deepOrange),
                    if (ward.ndvi != null)
                      _MetaChip(
                          icon: Icons.park,
                          label: 'NDVI ${ward.ndvi!.toStringAsFixed(2)}',
                          color: Colors.green),
                    if (ward.impervious != null)
                      _MetaChip(
                          icon: Icons.location_city,
                          label: 'Impervious ${ward.impervious!.toStringAsFixed(0)}%',
                          color: Colors.blueGrey),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 8),

            // Forecast chart
            const Text('12-Hour Forecast',
                style: TextStyle(fontWeight: FontWeight.w700, fontSize: 16)),
            const SizedBox(height: 8),
            forecastAsync.when(
              loading: () => const Center(child: CircularProgressIndicator()),
              error: (e, _) => Text('Forecast unavailable: $e',
                  style: const TextStyle(color: Colors.grey)),
              data: (forecasts) {
                final wardForecasts =
                    forecasts.where((f) => f.wardId == wardId).toList()
                      ..sort((a, b) => a.hour.compareTo(b.hour));
                if (wardForecasts.isEmpty) return const SizedBox.shrink();

                return Card(
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(8, 16, 16, 8),
                    child: SizedBox(
                      height: 180,
                      child: LineChart(
                        LineChartData(
                          gridData: FlGridData(
                            drawHorizontalLine: true,
                            horizontalInterval: 25,
                            getDrawingHorizontalLine: (_) => FlLine(
                              color: Colors.grey.shade200, strokeWidth: 1),
                          ),
                          titlesData: FlTitlesData(
                            leftTitles: AxisTitles(
                              sideTitles: SideTitles(
                                showTitles: true,
                                reservedSize: 32,
                                getTitlesWidget: (v, _) => Text(
                                  v.toInt().toString(),
                                  style: const TextStyle(fontSize: 10),
                                ),
                              ),
                            ),
                            bottomTitles: AxisTitles(
                              sideTitles: SideTitles(
                                showTitles: true,
                                interval: 2,
                                getTitlesWidget: (v, _) => Text(
                                  '+${v.toInt()}h',
                                  style: const TextStyle(fontSize: 9),
                                ),
                              ),
                            ),
                            topTitles: const AxisTitles(
                                sideTitles: SideTitles(showTitles: false)),
                            rightTitles: const AxisTitles(
                                sideTitles: SideTitles(showTitles: false)),
                          ),
                          borderData: FlBorderData(show: false),
                          minY: 0,
                          maxY: 100,
                          lineBarsData: [
                            LineChartBarData(
                              spots: wardForecasts
                                  .map((f) => FlSpot(
                                      f.hour.toDouble(),
                                      f.heatStressPredicted))
                                  .toList(),
                              color: Colors.deepOrange,
                              barWidth: 2,
                              dotData: const FlDotData(show: false),
                              belowBarData: BarAreaData(
                                show: true,
                                color: Colors.deepOrange.withOpacity(0.08),
                              ),
                            ),
                            LineChartBarData(
                              spots: wardForecasts
                                  .map((f) => FlSpot(
                                      f.hour.toDouble(),
                                      f.floodRiskPredicted))
                                  .toList(),
                              color: Colors.blue,
                              barWidth: 2,
                              dotData: const FlDotData(show: false),
                              belowBarData: BarAreaData(
                                show: true,
                                color: Colors.blue.withOpacity(0.08),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                );
              },
            ),

            const SizedBox(height: 8),
            // Legend
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: const [
                _LegendDot(color: Colors.deepOrange, label: 'Heat stress'),
                SizedBox(width: 16),
                _LegendDot(color: Colors.blue, label: 'Flood risk'),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _MetaChip extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  const _MetaChip({required this.icon, required this.label, required this.color});

  @override
  Widget build(BuildContext context) => Chip(
        avatar: Icon(icon, size: 14, color: color),
        label: Text(label, style: const TextStyle(fontSize: 12)),
        padding: const EdgeInsets.symmetric(horizontal: 4),
        visualDensity: VisualDensity.compact,
      );
}

class _LegendDot extends StatelessWidget {
  final Color color;
  final String label;
  const _LegendDot({required this.color, required this.label});

  @override
  Widget build(BuildContext context) => Row(
        children: [
          Container(width: 12, height: 3, color: color,
              decoration: BoxDecoration(borderRadius: BorderRadius.circular(2), color: color)),
          const SizedBox(width: 4),
          Text(label, style: const TextStyle(fontSize: 11, color: Colors.grey)),
        ],
      );
}
