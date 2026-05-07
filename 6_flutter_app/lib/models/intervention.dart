/// Intervention recommendation model matching FastAPI RecommendedIntervention.
class RecommendedIntervention {
  final int priorityRank;
  final int wardId;
  final String? wardName;
  final String interventionType;
  final double expectedHeatReduction;
  final double expectedFloodReduction;
  final int cost;
  final double wardHeatStress;
  final double wardFloodRisk;

  const RecommendedIntervention({
    required this.priorityRank,
    required this.wardId,
    this.wardName,
    required this.interventionType,
    required this.expectedHeatReduction,
    required this.expectedFloodReduction,
    required this.cost,
    required this.wardHeatStress,
    required this.wardFloodRisk,
  });

  factory RecommendedIntervention.fromJson(Map<String, dynamic> j) =>
      RecommendedIntervention(
        priorityRank: j['priority_rank'] as int,
        wardId: j['ward_id'] as int,
        wardName: j['ward_name'] as String?,
        interventionType: j['intervention_type'] as String,
        expectedHeatReduction: (j['expected_heat_reduction'] as num).toDouble(),
        expectedFloodReduction: (j['expected_flood_reduction'] as num).toDouble(),
        cost: j['cost'] as int,
        wardHeatStress: (j['ward_heat_stress'] as num).toDouble(),
        wardFloodRisk: (j['ward_flood_risk'] as num).toDouble(),
      );

  String get displayType => interventionType
      .replaceAll('_', ' ')
      .split(' ')
      .map((w) => w.isNotEmpty ? '${w[0].toUpperCase()}${w.substring(1)}' : '')
      .join(' ');
}

class InterventionResponse {
  final List<RecommendedIntervention> recommendations;
  final int budgetUsed;
  final int budgetRemaining;

  const InterventionResponse({
    required this.recommendations,
    required this.budgetUsed,
    required this.budgetRemaining,
  });

  factory InterventionResponse.fromJson(Map<String, dynamic> j) =>
      InterventionResponse(
        recommendations: (j['recommendations'] as List)
            .map((e) =>
                RecommendedIntervention.fromJson(e as Map<String, dynamic>))
            .toList(),
        budgetUsed: j['budget_used'] as int,
        budgetRemaining: j['budget_remaining'] as int,
      );
}
