import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/ward.dart';
import '../models/alert.dart';
import '../models/intervention.dart';
import '../services/api_client.dart';
import '../services/websocket_service.dart';

// ---------------------------------------------------------------------------
// Ward risks — auto-refreshes every 60 seconds
// ---------------------------------------------------------------------------

final wardRisksProvider =
    AsyncNotifierProvider<WardRisksNotifier, WardRiskResponse>(
  WardRisksNotifier.new,
);

class WardRisksNotifier extends AsyncNotifier<WardRiskResponse> {
  Timer? _refreshTimer;

  @override
  Future<WardRiskResponse> build() async {
    // Auto-refresh every 10 seconds for live feel
    _refreshTimer?.cancel();
    _refreshTimer = Timer.periodic(const Duration(seconds: 10), (_) {
      ref.invalidateSelf();
    });
    ref.onDispose(() => _refreshTimer?.cancel());
    return ApiClient.instance.getWardRisks();
  }

  Future<void> refresh() async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(
      () => ApiClient.instance.getWardRisks(),
    );
  }
}

// ---------------------------------------------------------------------------
// Selected ward filter
// ---------------------------------------------------------------------------

final riskLevelFilterProvider = StateProvider<String?>((ref) => null);

final filteredWardsProvider = Provider<AsyncValue<List<WardRisk>>>((ref) {
  final filter = ref.watch(riskLevelFilterProvider);
  return ref.watch(wardRisksProvider).whenData((resp) {
    if (filter == null) return resp.wards;
    return resp.wards.where((w) => w.riskLevel == filter).toList();
  });
});

// ---------------------------------------------------------------------------
// Ward detail + forecast
// ---------------------------------------------------------------------------

final wardDetailProvider =
    FutureProvider.family<WardRisk, int>((ref, wardId) async {
  return ApiClient.instance.getWardDetail(wardId);
});

final wardForecastProvider =
    FutureProvider.family<List<WardForecast>, int>((ref, wardId) async {
  return ApiClient.instance.getWardForecast(
    wardIds: [wardId],
    horizonHrs: 12,
  );
});

// ---------------------------------------------------------------------------
// Interventions
// ---------------------------------------------------------------------------

final interventionBudgetProvider = StateProvider<int>((ref) => 60);

final interventionResponseProvider =
    AsyncNotifierProvider<InterventionNotifier, InterventionResponse>(
  InterventionNotifier.new,
);

class InterventionNotifier extends AsyncNotifier<InterventionResponse> {
  @override
  Future<InterventionResponse> build() async {
    final budget = ref.watch(interventionBudgetProvider);
    return ApiClient.instance.getRecommendations(budget: budget);
  }

  Future<void> refresh({required int budget}) async {
    state = const AsyncLoading();
    state = await AsyncValue.guard(
      () => ApiClient.instance.getRecommendations(budget: budget, topK: 15),
    );
  }
}

// ---------------------------------------------------------------------------
// Live alerts (from WebSocket stream)
// ---------------------------------------------------------------------------

final alertsProvider = StreamProvider<List<WardAlert>>((ref) {
  final ws = ref.watch(wsServiceProvider);
  ws.connect();
  ref.onDispose(ws.disconnect);
  // WebSocketService accumulates internally — just return its stream directly
  return ws.alertStream;
});

// Derived: unread count for badge
final unreadAlertCountProvider = StateProvider<int>((ref) => 0);

final criticalWardsProvider = Provider<List<WardRisk>>((ref) {
  return ref.watch(filteredWardsProvider).whenOrNull(
        data: (wards) =>
            wards.where((w) => w.riskLevel == 'critical').toList()
              ..sort((a, b) => b.combinedRisk.compareTo(a.combinedRisk)),
      ) ??
      [];
});
