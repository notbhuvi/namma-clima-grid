import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/ward.dart';
import '../providers/ward_providers.dart';
import '../widgets/app_theme.dart';
import 'ward_detail_screen.dart';

// Live clock provider — ticks every second
final _clockProvider = StreamProvider<DateTime>((ref) {
  return Stream.periodic(const Duration(seconds: 1), (_) => DateTime.now());
});

class DashboardScreen extends ConsumerWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final wardsAsync = ref.watch(filteredWardsProvider);
    final filter = ref.watch(riskLevelFilterProvider);

    final clock = ref.watch(_clockProvider).valueOrNull ?? DateTime.now();
    final hh = clock.hour.toString().padLeft(2, '0');
    final mm = clock.minute.toString().padLeft(2, '0');
    final ss = clock.second.toString().padLeft(2, '0');

    return Scaffold(
      appBar: AppBar(
        title: const Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            NcgMark(size: 34),
            SizedBox(width: 10),
            Text('NammaClimaGrid'),
          ],
        ),
        actions: [
          // Live clock
          Center(
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 7,
                  height: 7,
                  decoration: BoxDecoration(
                    color: Colors.greenAccent.shade400,
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: Colors.greenAccent.withOpacity(0.6),
                        blurRadius: 5,
                      )
                    ],
                  ),
                ),
                const SizedBox(width: 5),
                Text(
                  '$hh:$mm:$ss',
                  style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 1,
                    color: Colors.white,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 4),
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => ref.read(wardRisksProvider.notifier).refresh(),
          ),
          const ThemeModeMenu(),
        ],
      ),
      body: NatureBackdrop(
        child: Column(
          children: [
            const _CitizenHero(),
            _SummaryBar(),
            _LiveBanner(),
            _FilterChips(selected: filter),
            Expanded(
              child: wardsAsync.when(
                loading: () => const Center(child: CircularProgressIndicator()),
                error: (e, _) => Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.wifi_off, size: 48, color: Colors.grey),
                      const SizedBox(height: 12),
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 24),
                        child: Text(
                          'Could not load ward data.\n$e',
                          textAlign: TextAlign.center,
                          style:
                              const TextStyle(color: Colors.grey, fontSize: 13),
                        ),
                      ),
                      const SizedBox(height: 16),
                      ElevatedButton(
                        onPressed: () =>
                            ref.read(wardRisksProvider.notifier).refresh(),
                        child: const Text('Retry'),
                      ),
                    ],
                  ),
                ),
                data: (wards) => _WardList(wards: wards),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _CitizenHero extends StatelessWidget {
  const _CitizenHero();

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final dark = Theme.of(context).brightness == Brightness.dark;
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.fromLTRB(14, 14, 14, 6),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(28),
        color: Theme.of(context).cardTheme.color,
        border: Border.all(color: scheme.primary.withOpacity(.08)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(dark ? .22 : .07),
            blurRadius: 32,
            offset: const Offset(0, 16),
          ),
        ],
      ),
      child: Row(
        children: [
          const NcgMark(size: 58),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Citizen Climate Pulse',
                  style: TextStyle(
                    color: scheme.onSurface,
                    fontSize: 22,
                    fontWeight: FontWeight.w900,
                    letterSpacing: -.4,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  'Track ward risk, map hotspots, report incidents, and receive BBMP alerts.',
                  style: TextStyle(
                    color: scheme.onSurfaceVariant,
                    fontSize: 12.5,
                    height: 1.25,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Summary bar
// ---------------------------------------------------------------------------

class _SummaryBar extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return ref.watch(wardRisksProvider).maybeWhen(
          data: (resp) {
            final wards = resp.wards;
            if (wards.isEmpty) return const SizedBox.shrink();
            final avgHeat =
                wards.map((w) => w.heatStressScore).reduce((a, b) => a + b) /
                    wards.length;
            final avgFlood =
                wards.map((w) => w.floodRiskScore).reduce((a, b) => a + b) /
                    wards.length;
            final nCritical =
                wards.where((w) => w.riskLevel == 'critical').length;
            final nHigh = wards.where((w) => w.riskLevel == 'high').length;

            return Container(
              margin: const EdgeInsets.fromLTRB(14, 14, 14, 8),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
              decoration: BoxDecoration(
                color: Theme.of(context).cardTheme.color,
                borderRadius: BorderRadius.circular(24),
                border: Border.all(
                    color:
                        Theme.of(context).colorScheme.primary.withOpacity(.08)),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(
                        Theme.of(context).brightness == Brightness.dark
                            ? .18
                            : .06),
                    blurRadius: 28,
                    offset: const Offset(0, 14),
                  ),
                ],
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _Stat(
                    icon: Icons.thermostat,
                    value: avgHeat.toStringAsFixed(0),
                    unit: '/100',
                    label: 'Avg Heat',
                    color: Colors.deepOrange,
                  ),
                  _divider(context),
                  _Stat(
                    icon: Icons.water,
                    value: avgFlood.toStringAsFixed(0),
                    unit: '/100',
                    label: 'Avg Flood',
                    color: Colors.blue,
                  ),
                  _divider(context),
                  _Stat(
                    icon: Icons.warning_amber,
                    value: '$nCritical',
                    unit: ' critical',
                    label: '$nHigh high',
                    color: Colors.red,
                  ),
                ],
              ),
            );
          },
          orElse: () => const SizedBox.shrink(),
        );
  }

  Widget _divider(BuildContext context) => Container(
        width: 1,
        height: 36,
        color: Theme.of(context).colorScheme.outlineVariant.withOpacity(0.8),
      );
}

class _Stat extends StatelessWidget {
  final IconData icon;
  final String value;
  final String unit;
  final String label;
  final Color color;

  const _Stat({
    required this.icon,
    required this.value,
    required this.unit,
    required this.label,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Row(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.baseline,
          textBaseline: TextBaseline.alphabetic,
          children: [
            Icon(icon, color: color, size: 16),
            const SizedBox(width: 4),
            Text(value,
                style: TextStyle(
                    fontWeight: FontWeight.w800, fontSize: 18, color: color)),
            Text(unit,
                style: TextStyle(fontSize: 11, color: color.withOpacity(0.8))),
          ],
        ),
        const SizedBox(height: 2),
        Text(label, style: const TextStyle(fontSize: 11, color: Colors.grey)),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Live data banner
// ---------------------------------------------------------------------------

class _LiveBanner extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return ref.watch(wardRisksProvider).maybeWhen(
          data: (resp) {
            final src = resp.source ?? '';
            final isLiveDb = src == 'postgresql_live';
            final isLiveWeather = src == 'live_weather';
            final dotColor = isLiveDb
                ? Colors.green
                : isLiveWeather
                    ? Colors.blue
                    : Colors.orange;
            final bgColor = isLiveDb
                ? Colors.green.shade50
                : isLiveWeather
                    ? Colors.blue.shade50
                    : Colors.orange.shade50;
            final label = isLiveDb
                ? '🟢 Live DB · Refreshes every 10s'
                : isLiveWeather
                    ? '🌤 Live Weather (Open-Meteo) · Refreshes every 10s'
                    : '🟡 Model data · Refreshes every 10s';
            return Container(
              width: double.infinity,
              margin: const EdgeInsets.symmetric(horizontal: 14, vertical: 4),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              decoration: BoxDecoration(
                color: bgColor.withOpacity(
                    Theme.of(context).brightness == Brightness.dark
                        ? .18
                        : .82),
                borderRadius: BorderRadius.circular(18),
              ),
              child: Row(
                children: [
                  Container(
                    width: 6,
                    height: 6,
                    decoration: BoxDecoration(
                      color: dotColor,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 6),
                  Text(
                    label,
                    style: TextStyle(
                      fontSize: 11,
                      color: dotColor.withOpacity(0.85),
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  const Spacer(),
                  Text(
                    'Updated ${TimeOfDay.now().format(context)}',
                    style: TextStyle(
                      fontSize: 11,
                      color: Colors.grey.shade500,
                    ),
                  ),
                ],
              ),
            );
          },
          orElse: () => const SizedBox.shrink(),
        );
  }
}

// ---------------------------------------------------------------------------
// Filter chips
// ---------------------------------------------------------------------------

class _FilterChips extends ConsumerWidget {
  final String? selected;
  const _FilterChips({required this.selected});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    const levels = ['critical', 'high', 'medium', 'low'];
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      child: Row(
        children: [
          _chip(
            context: context,
            label: 'All',
            isSelected: selected == null,
            color: const Color(0xFF1B6B3A),
            onTap: () =>
                ref.read(riskLevelFilterProvider.notifier).state = null,
          ),
          const SizedBox(width: 6),
          ...levels.map((lvl) => Padding(
                padding: const EdgeInsets.only(right: 6),
                child: _chip(
                  context: context,
                  label: lvl[0].toUpperCase() + lvl.substring(1),
                  isSelected: selected == lvl,
                  color: _levelColor(lvl),
                  onTap: () => ref
                      .read(riskLevelFilterProvider.notifier)
                      .state = selected == lvl ? null : lvl,
                ),
              )),
        ],
      ),
    );
  }

  Widget _chip({
    required BuildContext context,
    required String label,
    required bool isSelected,
    required Color color,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
        decoration: BoxDecoration(
          color: isSelected ? color : Colors.transparent,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
              color: isSelected ? color : Colors.grey.shade300, width: 1.5),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w600,
            color: isSelected
                ? Colors.white
                : Theme.of(context).colorScheme.onSurfaceVariant,
          ),
        ),
      ),
    );
  }

  Color _levelColor(String lvl) {
    switch (lvl) {
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
}

// ---------------------------------------------------------------------------
// Ward list — single column, compact rows
// ---------------------------------------------------------------------------

class _WardList extends StatelessWidget {
  final List<WardRisk> wards;
  const _WardList({required this.wards});

  @override
  Widget build(BuildContext context) {
    if (wards.isEmpty) {
      return const Center(
        child: Text('No wards match the selected filter.',
            style: TextStyle(color: Colors.grey)),
      );
    }
    return ListView.separated(
      padding: const EdgeInsets.fromLTRB(12, 4, 12, 16),
      itemCount: wards.length,
      separatorBuilder: (_, __) => const SizedBox(height: 6),
      itemBuilder: (ctx, i) {
        final ward = wards[i];
        return _WardRow(
          ward: ward,
          onTap: () => Navigator.of(ctx).push(MaterialPageRoute(
            builder: (_) => WardDetailScreen(wardId: ward.wardId),
          )),
        );
      },
    );
  }
}

// ---------------------------------------------------------------------------
// Single ward row card
// ---------------------------------------------------------------------------

class _WardRow extends StatelessWidget {
  final WardRisk ward;
  final VoidCallback onTap;
  const _WardRow({required this.ward, required this.onTap});

  Color get _accentColor {
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
    final scheme = Theme.of(context).colorScheme;
    return Material(
      color: Theme.of(context).cardTheme.color,
      borderRadius: BorderRadius.circular(20),
      child: InkWell(
        borderRadius: BorderRadius.circular(20),
        onTap: onTap,
        child: Container(
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(20),
            border: Border(
              left: BorderSide(color: _accentColor, width: 4),
            ),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(
                    Theme.of(context).brightness == Brightness.dark
                        ? .18
                        : .06),
                blurRadius: 22,
                offset: const Offset(0, 10),
              ),
            ],
          ),
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          child: Row(
            children: [
              // Ward name + LST
              Expanded(
                flex: 5,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      ward.displayName,
                      style: TextStyle(
                          fontWeight: FontWeight.w900,
                          fontSize: 14,
                          color: scheme.onSurface),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    if (ward.temperatureC != null ||
                        ward.lstPredCelsius != null) ...[
                      const SizedBox(height: 2),
                      Row(
                        children: [
                          Text(
                            ward.temperatureC != null
                                ? '${ward.temperatureC!.toStringAsFixed(1)}°C'
                                : '${ward.lstPredCelsius!.toStringAsFixed(1)}°C',
                            style: TextStyle(
                                fontSize: 11,
                                color: ward.temperatureC != null
                                    ? Colors.blue.shade700
                                    : Colors.grey.shade600,
                                fontWeight: ward.temperatureC != null
                                    ? FontWeight.w600
                                    : FontWeight.normal),
                          ),
                          if (ward.rainfallMm != null &&
                              ward.rainfallMm! > 0) ...[
                            const SizedBox(width: 4),
                            Text(
                              '💧${ward.rainfallMm!.toStringAsFixed(1)}mm',
                              style: TextStyle(
                                  fontSize: 10, color: Colors.blue.shade400),
                            ),
                          ],
                        ],
                      ),
                    ],
                  ],
                ),
              ),
              const SizedBox(width: 10),
              // Score bars
              Expanded(
                flex: 6,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    _ScoreBar(
                      label: '🌡',
                      value: ward.heatStressScore,
                      color: _heatColor(ward.heatStressScore),
                    ),
                    const SizedBox(height: 4),
                    _ScoreBar(
                      label: '💧',
                      value: ward.floodRiskScore,
                      color: Colors.blue.shade400,
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 10),
              // Risk badge
              _RiskBadge(level: ward.riskLevel, color: _accentColor),
            ],
          ),
        ),
      ),
    );
  }

  Color _heatColor(double v) {
    if (v >= 75) return const Color(0xFFD32F2F);
    if (v >= 50) return const Color(0xFFF57C00);
    if (v >= 25) return const Color(0xFFFBC02D);
    return const Color(0xFF388E3C);
  }
}

class _ScoreBar extends StatelessWidget {
  final String label;
  final double value;
  final Color color;
  const _ScoreBar(
      {required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Text(label, style: const TextStyle(fontSize: 11)),
        const SizedBox(width: 4),
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: value / 100,
              minHeight: 6,
              backgroundColor: Colors.grey.shade200,
              valueColor: AlwaysStoppedAnimation<Color>(color),
            ),
          ),
        ),
        const SizedBox(width: 5),
        SizedBox(
          width: 26,
          child: Text(
            value.toStringAsFixed(0),
            style: TextStyle(
                fontSize: 11, fontWeight: FontWeight.w600, color: color),
            textAlign: TextAlign.right,
          ),
        ),
      ],
    );
  }
}

class _RiskBadge extends StatelessWidget {
  final String level;
  final Color color;
  const _RiskBadge({required this.level, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(
        level.toUpperCase(),
        style: TextStyle(
          color: color,
          fontSize: 9,
          fontWeight: FontWeight.w800,
          letterSpacing: 0.5,
        ),
      ),
    );
  }
}
