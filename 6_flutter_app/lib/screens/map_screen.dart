import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:latlong2/latlong.dart';

import '../services/base_url.dart';
import 'ward_detail_screen.dart';

// ────────────────────────────────────────────────────────────────────────────
// Providers
// ────────────────────────────────────────────────────────────────────────────

final _geoJsonProvider =
    FutureProvider<List<Map<String, dynamic>>>((ref) async {
  final dio = Dio(BaseOptions(
    baseUrl: apiBaseUrl,
    connectTimeout: const Duration(seconds: 10),
    receiveTimeout: const Duration(seconds: 15),
  ));
  final resp = await dio.get('/wards/geojson/live');
  final body = resp.data as Map<String, dynamic>;
  return (body['features'] as List).cast<Map<String, dynamic>>();
});

enum _MapLayer { heat, flood, combined }

final _mapLayerProvider = StateProvider<_MapLayer>((_) => _MapLayer.heat);
final _selectedWardProvider =
    StateProvider<Map<String, dynamic>?>((ref) => null);

// ────────────────────────────────────────────────────────────────────────────
// Screen
// ────────────────────────────────────────────────────────────────────────────

class MapScreen extends ConsumerWidget {
  const MapScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final geoAsync = ref.watch(_geoJsonProvider);
    final layer = ref.watch(_mapLayerProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Ward Risk Map'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            tooltip: 'Refresh',
            onPressed: () {
              ref.invalidate(_geoJsonProvider);
              ref.read(_selectedWardProvider.notifier).state = null;
            },
          ),
        ],
      ),
      body: Column(
        children: [
          _LayerToggle(selected: layer),
          Expanded(
            child: geoAsync.when(
              loading: () => const Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 12),
                    Text('Loading ward map…',
                        style: TextStyle(color: Colors.grey)),
                  ],
                ),
              ),
              error: (e, _) => _MapError(error: e.toString()),
              data: (features) => _LiveMap(features: features, layer: layer),
            ),
          ),
        ],
      ),
    );
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Layer toggle
// ────────────────────────────────────────────────────────────────────────────

Color _layerColor(_MapLayer l) {
  switch (l) {
    case _MapLayer.heat:     return Colors.deepOrange;
    case _MapLayer.flood:    return Colors.blue;
    case _MapLayer.combined: return Colors.purple;
  }
}

class _LayerToggle extends ConsumerWidget {
  final _MapLayer selected;
  const _LayerToggle({required this.selected});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Container(
      color: Theme.of(context).colorScheme.surface,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      child: Row(
        children: [
          const Text('Color by:',
              style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
          const SizedBox(width: 8),
          ..._MapLayer.values.map((l) {
            final label = switch (l) {
              _MapLayer.heat     => 'Heat',
              _MapLayer.flood    => 'Flood',
              _MapLayer.combined => 'Combined',
            };
            return Padding(
              padding: const EdgeInsets.only(right: 6),
              child: ChoiceChip(
                label: Text(label, style: const TextStyle(fontSize: 12)),
                selected: selected == l,
                selectedColor: _layerColor(l).withOpacity(0.2),
                onSelected: (_) =>
                    ref.read(_mapLayerProvider.notifier).state = l,
              ),
            );
          }),
          const Spacer(),
          _GradientLegend(layer: selected),
        ],
      ),
    );
  }
}

class _GradientLegend extends StatelessWidget {
  final _MapLayer layer;
  const _GradientLegend({required this.layer});

  @override
  Widget build(BuildContext context) {
    final c = _layerColor(layer);
    return Row(
      children: [
        const Text('Low', style: TextStyle(fontSize: 10, color: Colors.grey)),
        const SizedBox(width: 4),
        Container(
          width: 56,
          height: 10,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(4),
            gradient: LinearGradient(colors: [c.withOpacity(0.15), c]),
          ),
        ),
        const SizedBox(width: 4),
        const Text('High', style: TextStyle(fontSize: 10, color: Colors.grey)),
      ],
    );
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Map widget
// ────────────────────────────────────────────────────────────────────────────

class _LiveMap extends ConsumerStatefulWidget {
  final List<Map<String, dynamic>> features;
  final _MapLayer layer;

  const _LiveMap({required this.features, required this.layer});

  @override
  ConsumerState<_LiveMap> createState() => _LiveMapState();
}

class _LiveMapState extends ConsumerState<_LiveMap> {
  final MapController _mapController = MapController();

  @override
  void dispose() {
    _mapController.dispose();
    super.dispose();
  }

  // ── Point-in-polygon (ray casting) ──────────────────────────────────────

  bool _pointInPolygon(LatLng point, List<LatLng> polygon) {
    bool inside = false;
    int j = polygon.length - 1;
    for (int i = 0; i < polygon.length; i++) {
      final xi = polygon[i].longitude, yi = polygon[i].latitude;
      final xj = polygon[j].longitude, yj = polygon[j].latitude;
      final intersect = ((yi > point.latitude) != (yj > point.latitude)) &&
          (point.longitude < (xj - xi) * (point.latitude - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
      j = i;
    }
    return inside;
  }

  Map<String, dynamic>? _findTapped(LatLng point) {
    for (final f in widget.features) {
      final geom = f['geometry'] as Map<String, dynamic>? ?? {};
      if (geom['type'] != 'Polygon') continue;
      final rings = (geom['coordinates'] as List).first as List;
      final pts = rings.map((c) {
        final coord = c as List;
        return LatLng((coord[1] as num).toDouble(),
            (coord[0] as num).toDouble());
      }).toList();
      if (_pointInPolygon(point, pts)) return f;
    }
    return null;
  }

  // ── Score and color helpers ──────────────────────────────────────────────

  double _score(Map<String, dynamic> p) {
    final heat = (p['heat_stress_score'] as num?)?.toDouble() ?? 0;
    final flood = (p['flood_risk_score'] as num?)?.toDouble() ?? 0;
    return switch (widget.layer) {
      _MapLayer.heat     => heat,
      _MapLayer.flood    => flood,
      _MapLayer.combined => 0.6 * heat + 0.4 * flood,
    };
  }

  Color _scoreColor(double score) {
    final t = (score / 100).clamp(0.0, 1.0);
    final base = _layerColor(widget.layer);
    return Color.lerp(base.withOpacity(0.08), base, t)!;
  }

  @override
  Widget build(BuildContext context) {
    final selected = ref.watch(_selectedWardProvider);

    final polygons = widget.features.map((f) {
      final props = f['properties'] as Map<String, dynamic>? ?? {};
      final geom = f['geometry'] as Map<String, dynamic>? ?? {};
      if (geom['type'] != 'Polygon') return null;

      final rings = (geom['coordinates'] as List).first as List;
      final pts = rings.map((c) {
        final coord = c as List;
        return LatLng((coord[1] as num).toDouble(),
            (coord[0] as num).toDouble());
      }).toList();

      final sc = _score(props);
      final col = _scoreColor(sc);
      final isSelected = selected != null &&
          (selected['properties']?['ward_id']) == props['ward_id'];

      return Polygon(
        points: pts,
        color: col.withOpacity(isSelected ? 0.75 : 0.55),
        borderColor: isSelected ? Colors.white : col,
        borderStrokeWidth: isSelected ? 2.5 : 1.0,
        isFilled: true,
      );
    }).whereType<Polygon>().toList();

    return Stack(
      children: [
        FlutterMap(
          mapController: _mapController,
          options: MapOptions(
            initialCenter: const LatLng(12.9716, 77.5946),
            initialZoom: 11.5,
            maxZoom: 16,
            minZoom: 9,
            onTap: (tapPos, latlng) {
              final hit = _findTapped(latlng);
              ref.read(_selectedWardProvider.notifier).state = hit;
            },
          ),
          children: [
            TileLayer(
              urlTemplate: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
              userAgentPackageName: 'com.ncg.namma_clima_grid',
              maxZoom: 19,
            ),
            PolygonLayer(polygons: polygons, polygonCulling: true),
          ],
        ),
        // Tooltip overlay
        if (selected != null)
          Positioned(
            bottom: 16,
            left: 16,
            right: 16,
            child: _WardTooltip(
              props: selected['properties'] as Map<String, dynamic>? ?? {},
              onClose: () =>
                  ref.read(_selectedWardProvider.notifier).state = null,
              onDetail: (wardId) {
                ref.read(_selectedWardProvider.notifier).state = null;
                Navigator.of(context).push(MaterialPageRoute(
                  builder: (_) => WardDetailScreen(wardId: wardId),
                ));
              },
            ),
          ),
        // Ward count badge
        Positioned(
          top: 10,
          right: 10,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.6),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              '${widget.features.length} wards',
              style: const TextStyle(
                  color: Colors.white,
                  fontSize: 11,
                  fontWeight: FontWeight.w600),
            ),
          ),
        ),
      ],
    );
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Ward tooltip card
// ────────────────────────────────────────────────────────────────────────────

class _WardTooltip extends StatelessWidget {
  final Map<String, dynamic> props;
  final VoidCallback onClose;
  final void Function(int) onDetail;

  const _WardTooltip({
    required this.props,
    required this.onClose,
    required this.onDetail,
  });

  @override
  Widget build(BuildContext context) {
    final wardId = (props['ward_id'] as num?)?.toInt() ?? 0;
    final name = props['ward_name'] as String? ?? 'Ward $wardId';
    final heat = (props['heat_stress_score'] as num?)?.toDouble() ?? 0;
    final flood = (props['flood_risk_score'] as num?)?.toDouble() ?? 0;
    final riskLevel = props['risk_level'] as String? ?? 'unknown';

    final riskColor = switch (riskLevel) {
      'critical' => const Color(0xFFD32F2F),
      'high'     => const Color(0xFFF57C00),
      'medium'   => const Color(0xFFFBC02D),
      _          => const Color(0xFF388E3C),
    };

    return Material(
      elevation: 8,
      borderRadius: BorderRadius.circular(14),
      child: Padding(
        padding: const EdgeInsets.fromLTRB(14, 10, 14, 12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(name,
                      style: const TextStyle(
                          fontWeight: FontWeight.w800, fontSize: 15)),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: riskColor.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: riskColor),
                  ),
                  child: Text(
                    riskLevel.toUpperCase(),
                    style: TextStyle(
                        color: riskColor,
                        fontSize: 10,
                        fontWeight: FontWeight.w700),
                  ),
                ),
                const SizedBox(width: 6),
                GestureDetector(
                  onTap: onClose,
                  child: const Icon(Icons.close, size: 18, color: Colors.grey),
                ),
              ],
            ),
            const SizedBox(height: 10),
            Row(
              children: [
                Expanded(
                  child: _ScoreBar(
                      label: 'Heat Stress',
                      value: heat,
                      color: Colors.deepOrange,
                      icon: Icons.thermostat),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: _ScoreBar(
                      label: 'Flood Risk',
                      value: flood,
                      color: Colors.blue,
                      icon: Icons.water),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                if (props['lst_pred_celsius'] != null)
                  _InfoChip(
                      label: 'LST ${(props['lst_pred_celsius'] as num).toStringAsFixed(1)}°C'),
                if (props['ndvi'] != null)
                  _InfoChip(
                      label: 'NDVI ${(props['ndvi'] as num).toStringAsFixed(2)}'),
                const Spacer(),
                TextButton.icon(
                  icon: const Icon(Icons.open_in_new, size: 14),
                  label: const Text('Details',
                      style: TextStyle(fontSize: 12)),
                  style: TextButton.styleFrom(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 10, vertical: 4),
                    tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  ),
                  onPressed: () => onDetail(wardId),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _ScoreBar extends StatelessWidget {
  final String label;
  final double value;
  final Color color;
  final IconData icon;
  const _ScoreBar(
      {required this.label,
      required this.value,
      required this.color,
      required this.icon});

  @override
  Widget build(BuildContext context) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            Icon(icon, size: 12, color: color),
            const SizedBox(width: 4),
            Text(label,
                style: TextStyle(
                    fontSize: 11,
                    color: color,
                    fontWeight: FontWeight.w600)),
            const Spacer(),
            Text('${value.toStringAsFixed(0)}/100',
                style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: color)),
          ]),
          const SizedBox(height: 4),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: value / 100,
              backgroundColor: color.withOpacity(0.1),
              valueColor: AlwaysStoppedAnimation(color),
              minHeight: 6,
            ),
          ),
        ],
      );
}

class _InfoChip extends StatelessWidget {
  final String label;
  const _InfoChip({required this.label});

  @override
  Widget build(BuildContext context) => Container(
        margin: const EdgeInsets.only(right: 6),
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
        decoration: BoxDecoration(
          color: Colors.grey.shade100,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Text(label,
            style: const TextStyle(fontSize: 10, color: Colors.grey)),
      );
}

// ────────────────────────────────────────────────────────────────────────────
// Error state
// ────────────────────────────────────────────────────────────────────────────

class _MapError extends ConsumerWidget {
  final String error;
  const _MapError({required this.error});

  @override
  Widget build(BuildContext context, WidgetRef ref) => Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.map_outlined, size: 56, color: Colors.grey),
            const SizedBox(height: 12),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 32),
              child: Text(
                'Could not load map data.\n$error',
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.grey),
              ),
            ),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              icon: const Icon(Icons.refresh),
              label: const Text('Retry'),
              onPressed: () => ref.invalidate(_geoJsonProvider),
            ),
          ],
        ),
      );
}
