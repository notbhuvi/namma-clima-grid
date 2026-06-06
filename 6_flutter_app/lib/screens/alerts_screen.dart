import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../models/alert.dart';
import '../providers/ward_providers.dart';
import '../widgets/app_theme.dart';
import '../widgets/alert_banner.dart';

class AlertsScreen extends ConsumerStatefulWidget {
  const AlertsScreen({super.key});

  @override
  ConsumerState<AlertsScreen> createState() => _AlertsScreenState();
}

class _AlertsScreenState extends ConsumerState<AlertsScreen> {
  final List<WardAlert> _dismissed = [];

  @override
  Widget build(BuildContext context) {
    final alertsAsync = ref.watch(alertsProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Live Alerts'),
        actions: [
          const ThemeModeMenu(),
          alertsAsync.maybeWhen(
            data: (alerts) {
              final active =
                  alerts.where((a) => !_dismissed.contains(a)).toList();
              if (active.isEmpty) return const SizedBox.shrink();
              return Padding(
                padding: const EdgeInsets.only(right: 12),
                child: Center(
                  child: Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                    decoration: BoxDecoration(
                      color: Colors.red,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Text(
                      '${active.length}',
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 12,
                          fontWeight: FontWeight.w700),
                    ),
                  ),
                ),
              );
            },
            orElse: () => const SizedBox.shrink(),
          ),
        ],
      ),
      body: NatureBackdrop(
        child: alertsAsync.when(
          loading: () => const Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                CircularProgressIndicator(),
                SizedBox(height: 12),
                Text('Connecting to alert stream…',
                    style: TextStyle(color: Colors.grey)),
              ],
            ),
          ),
          error: (e, _) => Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.wifi_off, size: 48, color: Colors.grey),
                const SizedBox(height: 12),
                Text('WebSocket disconnected.\n$e',
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: Colors.grey)),
              ],
            ),
          ),
          data: (alerts) {
            final active = alerts.where((a) => !_dismissed.contains(a)).toList()
              ..sort((a, b) => b.timestamp.compareTo(a.timestamp));

            if (active.isEmpty) {
              return Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: const [
                    Icon(Icons.check_circle_outline,
                        size: 56, color: Colors.green),
                    SizedBox(height: 12),
                    Text('All clear — no active alerts.',
                        style: TextStyle(color: Colors.grey, fontSize: 15)),
                  ],
                ),
              );
            }

            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
                  child: Text(
                    '${active.length} active alert${active.length > 1 ? 's' : ''} · '
                    'Updated ${DateFormat('HH:mm').format(DateTime.now())}',
                    style: const TextStyle(fontSize: 12, color: Colors.grey),
                  ),
                ),
                Expanded(
                  child: ListView.builder(
                    itemCount: active.length,
                    itemBuilder: (ctx, i) => AlertBanner(
                      alert: active[i],
                      onDismiss: () =>
                          setState(() => _dismissed.add(active[i])),
                    ),
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}
