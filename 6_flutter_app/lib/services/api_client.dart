import 'package:dio/dio.dart';

import '../models/ward.dart';
import '../models/intervention.dart';
import 'base_url.dart';

class ApiClient {
  ApiClient._()
      : _dio = Dio(BaseOptions(
          baseUrl: apiBaseUrl,
          connectTimeout: const Duration(seconds: 10),
          receiveTimeout: const Duration(seconds: 30),
          headers: {'Content-Type': 'application/json'},
        )) {
    _dio.interceptors.add(LogInterceptor(requestBody: false, responseBody: false));
  }

  static final ApiClient instance = ApiClient._();
  final Dio _dio;

  // ── Wards ──────────────────────────────────────────────────────────

  Future<WardRiskResponse> getWardRisks({
    String? riskLevel,
    String sortBy = 'heat_stress_score',
    int limit = 198,
  }) async {
    final resp = await _dio.get('/wards/risk', queryParameters: {
      if (riskLevel != null) 'risk_level': riskLevel,
      'sort_by': sortBy,
      'limit': limit,
    });
    return WardRiskResponse.fromJson(resp.data as Map<String, dynamic>);
  }

  Future<WardRisk> getWardDetail(int wardId) async {
    final resp = await _dio.get('/wards/risk/$wardId');
    final data = resp.data as Map<String, dynamic>;
    return WardRisk.fromJson(data['ward'] as Map<String, dynamic>);
  }

  Future<List<WardForecast>> getWardForecast({
    List<int>? wardIds,
    int horizonHrs = 6,
  }) async {
    final resp = await _dio.post('/wards/forecast', data: {
      if (wardIds != null) 'ward_ids': wardIds,
      'horizon_hrs': horizonHrs,
    });
    final body = resp.data as Map<String, dynamic>;
    return (body['forecasts'] as List)
        .map((e) => WardForecast.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  // ── Interventions ──────────────────────────────────────────────────

  Future<InterventionResponse> getRecommendations({
    int budget = 60,
    int topK = 10,
    List<int>? wardIds,
  }) async {
    final resp = await _dio.post('/interventions/recommend', data: {
      'budget': budget,
      'top_k': topK,
      if (wardIds != null) 'ward_ids': wardIds,
    });
    return InterventionResponse.fromJson(resp.data as Map<String, dynamic>);
  }

  Future<List<Map<String, dynamic>>> getInterventionTypes() async {
    final resp = await _dio.get('/interventions/types');
    return (resp.data as List).cast<Map<String, dynamic>>();
  }

  // ── Health ─────────────────────────────────────────────────────────

  Future<Map<String, dynamic>> getHealth() async {
    final resp = await _dio.get('/health');
    return resp.data as Map<String, dynamic>;
  }
}
