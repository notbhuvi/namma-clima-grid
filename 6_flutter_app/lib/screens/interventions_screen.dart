import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/intervention.dart';
import '../providers/ward_providers.dart';

class InterventionsScreen extends ConsumerWidget {
  const InterventionsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final budget   = ref.watch(interventionBudgetProvider);
    final recsAsync = ref.watch(interventionResponseProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Green Interventions'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () => ref
                .read(interventionResponseProvider.notifier)
                .refresh(budget: budget),
          ),
        ],
      ),
      body: Column(
        children: [
          // Budget slider
          _BudgetSlider(budget: budget),
          const Divider(height: 1),
          // Recommendations list
          Expanded(
            child: recsAsync.when(
              loading: () => const Center(child: CircularProgressIndicator()),
              error: (e, _) => Center(
                child: Text('Error: $e', style: const TextStyle(color: Colors.grey)),
              ),
              data: (resp) => _RecommendationsList(response: resp),
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Budget slider
// ---------------------------------------------------------------------------

class _BudgetSlider extends ConsumerWidget {
  final int budget;
  const _BudgetSlider({required this.budget});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text('Budget', style: TextStyle(fontWeight: FontWeight.w700)),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
                decoration: BoxDecoration(
                  color: const Color(0xFF1B6B3A).withOpacity(0.12),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  '$budget units',
                  style: const TextStyle(
                    color: Color(0xFF1B6B3A),
                    fontWeight: FontWeight.w700,
                    fontSize: 13,
                  ),
                ),
              ),
            ],
          ),
          Slider(
            value: budget.toDouble(),
            min: 10,
            max: 200,
            divisions: 19,
            activeColor: const Color(0xFF1B6B3A),
            label: '$budget',
            onChanged: (v) =>
                ref.read(interventionBudgetProvider.notifier).state = v.toInt(),
            onChangeEnd: (v) => ref
                .read(interventionResponseProvider.notifier)
                .refresh(budget: v.toInt()),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Recommendations list
// ---------------------------------------------------------------------------

class _RecommendationsList extends StatelessWidget {
  final InterventionResponse response;
  const _RecommendationsList({required this.response});

  @override
  Widget build(BuildContext context) {
    if (response.recommendations.isEmpty) {
      return const Center(
          child: Text('No recommendations available.',
              style: TextStyle(color: Colors.grey)));
    }
    return Column(
      children: [
        // Budget summary bar
        Container(
          color: Colors.grey.shade50,
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('${response.recommendations.length} recommendations',
                  style: const TextStyle(fontSize: 13, color: Colors.grey)),
              Text(
                'Used: ${response.budgetUsed}  ·  Left: ${response.budgetRemaining}',
                style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600),
              ),
            ],
          ),
        ),
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.only(bottom: 16),
            itemCount: response.recommendations.length,
            itemBuilder: (ctx, i) =>
                _InterventionTile(rec: response.recommendations[i]),
          ),
        ),
      ],
    );
  }
}

class _InterventionTile extends StatelessWidget {
  final RecommendedIntervention rec;
  const _InterventionTile({required this.rec});

  IconData get _icon {
    switch (rec.interventionType) {
      case 'tree_planting':      return Icons.park;
      case 'green_roof':         return Icons.roofing;
      case 'permeable_pavement': return Icons.texture;
      case 'urban_wetland':      return Icons.water;
      case 'cool_pavement':      return Icons.wb_sunny_outlined;
      default:                   return Icons.eco;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: const Color(0xFF1B6B3A).withOpacity(0.12),
          child: Icon(_icon, color: const Color(0xFF1B6B3A), size: 20),
        ),
        title: Row(
          children: [
            Container(
              width: 22,
              height: 22,
              decoration: BoxDecoration(
                color: const Color(0xFF1B6B3A),
                shape: BoxShape.circle,
              ),
              alignment: Alignment.center,
              child: Text(
                '${rec.priorityRank}',
                style: const TextStyle(color: Colors.white, fontSize: 11,
                    fontWeight: FontWeight.w700),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Text(rec.displayType,
                  style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 14)),
            ),
          ],
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(rec.wardName ?? 'Ward ${rec.wardId}',
                style: const TextStyle(fontSize: 12)),
            const SizedBox(height: 4),
            Row(
              children: [
                if (rec.expectedHeatReduction > 0)
                  _Delta(
                    icon: Icons.thermostat,
                    value: '-${rec.expectedHeatReduction.toStringAsFixed(1)}',
                    color: Colors.deepOrange,
                  ),
                if (rec.expectedFloodReduction > 0)
                  Padding(
                    padding: const EdgeInsets.only(left: 8),
                    child: _Delta(
                      icon: Icons.water,
                      value: '-${rec.expectedFloodReduction.toStringAsFixed(1)}',
                      color: Colors.blue,
                    ),
                  ),
              ],
            ),
          ],
        ),
        trailing: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.monetization_on_outlined, size: 14, color: Colors.grey),
            Text('${rec.cost}',
                style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 13)),
          ],
        ),
        isThreeLine: true,
      ),
    );
  }
}

class _Delta extends StatelessWidget {
  final IconData icon;
  final String value;
  final Color color;
  const _Delta({required this.icon, required this.value, required this.color});

  @override
  Widget build(BuildContext context) => Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 12, color: color),
          const SizedBox(width: 2),
          Text(value, style: TextStyle(fontSize: 11, color: color,
              fontWeight: FontWeight.w600)),
        ],
      );
}
