/// Ward risk data model matching FastAPI WardRisk schema.
class WardRisk {
  final int wardId;
  final String? wardName;
  final double heatStressScore;
  final double floodRiskScore;
  final String riskLevel;
  final double? lstPredCelsius;
  final double? ndvi;
  final double? impervious;
  final DateTime? updatedAt;
  // Live weather fields (present when source == "live_weather")
  final double? temperatureC;
  final double? rainfallMm;
  final double? humidityPct;
  final String? dataSource;
  final String? confidenceTier;
  final double? confidenceScore;
  final String? confidenceReason;

  const WardRisk({
    required this.wardId,
    this.wardName,
    required this.heatStressScore,
    required this.floodRiskScore,
    required this.riskLevel,
    this.lstPredCelsius,
    this.ndvi,
    this.impervious,
    this.updatedAt,
    this.temperatureC,
    this.rainfallMm,
    this.humidityPct,
    this.dataSource,
    this.confidenceTier,
    this.confidenceScore,
    this.confidenceReason,
  });

  factory WardRisk.fromJson(Map<String, dynamic> j) => WardRisk(
        wardId: j['ward_id'] as int,
        wardName: j['ward_name'] as String?,
        heatStressScore: (j['heat_stress_score'] as num).toDouble(),
        floodRiskScore: (j['flood_risk_score'] as num).toDouble(),
        riskLevel: j['risk_level'] as String,
        lstPredCelsius: (j['lst_pred_celsius'] as num?)?.toDouble(),
        ndvi: (j['ndvi'] as num?)?.toDouble(),
        impervious: (j['impervious_pct'] as num?)?.toDouble(),
        updatedAt: j['updated_at'] != null
            ? DateTime.tryParse(j['updated_at'] as String)
            : null,
        temperatureC: (j['temperature_c'] as num?)?.toDouble(),
        rainfallMm:   (j['rainfall_mm']   as num?)?.toDouble(),
        humidityPct:  (j['humidity_pct']  as num?)?.toDouble(),
        dataSource: j['data_source'] as String?,
        confidenceTier: j['confidence_tier'] as String?,
        confidenceScore: (j['confidence_score'] as num?)?.toDouble(),
        confidenceReason: j['confidence_reason'] as String?,
      );

  /// Combined risk score (0–100) used for sorting and colour mapping.
  double get combinedRisk => 0.6 * heatStressScore + 0.4 * floodRiskScore;

  String get displayName => wardName ?? 'Ward $wardId';
}

class WardRiskResponse {
  final List<WardRisk> wards;
  final int total;
  final DateTime timestamp;
  final String? source;
  final Map<String, dynamic>? predictionConfidence;

  const WardRiskResponse({
    required this.wards,
    required this.total,
    required this.timestamp,
    this.source,
    this.predictionConfidence,
  });

  factory WardRiskResponse.fromJson(Map<String, dynamic> j) =>
      WardRiskResponse(
        wards: (j['wards'] as List)
            .map((e) => WardRisk.fromJson(e as Map<String, dynamic>))
            .toList(),
        total: j['total'] as int,
        timestamp: DateTime.parse(j['timestamp'] as String),
        source: j['source'] as String?,
        predictionConfidence:
            j['prediction_confidence'] as Map<String, dynamic>?,
      );
}

class WardForecast {
  final int wardId;
  final int hour;
  final double heatStressPredicted;
  final double floodRiskPredicted;
  final double confidence;

  const WardForecast({
    required this.wardId,
    required this.hour,
    required this.heatStressPredicted,
    required this.floodRiskPredicted,
    required this.confidence,
  });

  factory WardForecast.fromJson(Map<String, dynamic> j) => WardForecast(
        wardId: j['ward_id'] as int,
        hour: j['hour'] as int,
        heatStressPredicted: (j['heat_stress_predicted'] as num).toDouble(),
        floodRiskPredicted: (j['flood_risk_predicted'] as num).toDouble(),
        confidence: (j['confidence'] as num).toDouble(),
      );
}
