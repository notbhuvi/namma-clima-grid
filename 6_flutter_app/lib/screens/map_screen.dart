import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:latlong2/latlong.dart';

import '../services/base_url.dart';
import '../widgets/app_theme.dart';
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

const double _heatMinC = 22.0;
const double _heatMaxC = 34.0;
const double _floodMin = 2.0;
const double _floodMax = 6.0;

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
          const ThemeModeMenu(),
        ],
      ),
      body: NatureBackdrop(
        child: Column(
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
      ),
    );
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Layer toggle
// ────────────────────────────────────────────────────────────────────────────

Color _layerColor(_MapLayer l) {
  switch (l) {
    case _MapLayer.heat:
      return const Color(0xFFD63C31);
    case _MapLayer.flood:
      return const Color(0xFF176BC2);
    case _MapLayer.combined:
      return const Color(0xFF0F5F37);
  }
}

class _LayerPalette {
  final Color low;
  final Color high;
  const _LayerPalette({required this.low, required this.high});
}

_LayerPalette _layerPalette(_MapLayer layer) {
  return switch (layer) {
    _MapLayer.heat =>
      const _LayerPalette(low: Color(0xFFFFEBD1), high: Color(0xFFD63C31)),
    _MapLayer.flood =>
      const _LayerPalette(low: Color(0xFFDAEDFF), high: Color(0xFF176BC2)),
    _MapLayer.combined =>
      const _LayerPalette(low: Color(0xFFE8F4EA), high: Color(0xFF0F5F37)),
  };
}

class _LayerToggle extends ConsumerWidget {
  final _MapLayer selected;
  const _LayerToggle({required this.selected});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Container(
      margin: const EdgeInsets.fromLTRB(12, 12, 12, 8),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Theme.of(context).cardTheme.color,
        borderRadius: BorderRadius.circular(22),
      ),
      child: Row(
        children: [
          const Text('Color by:',
              style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
          const SizedBox(width: 8),
          ..._MapLayer.values.map((l) {
            final label = switch (l) {
              _MapLayer.heat => 'Heat',
              _MapLayer.flood => 'Flood',
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
    final palette = _layerPalette(layer);
    final low = switch (layer) {
      _MapLayer.heat => '${_heatMinC.toStringAsFixed(0)}°C',
      _MapLayer.flood => _floodMin.toStringAsFixed(1),
      _MapLayer.combined => '0.0',
    };
    final high = switch (layer) {
      _MapLayer.heat => '${_heatMaxC.toStringAsFixed(0)}°C',
      _MapLayer.flood => _floodMax.toStringAsFixed(1),
      _MapLayer.combined => '1.0',
    };
    return Row(
      children: [
        Text(low, style: const TextStyle(fontSize: 10, color: Colors.grey)),
        const SizedBox(width: 4),
        Container(
          width: 56,
          height: 10,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(4),
            gradient: LinearGradient(colors: [palette.low, palette.high]),
          ),
        ),
        const SizedBox(width: 4),
        Text(high, style: const TextStyle(fontSize: 10, color: Colors.grey)),
      ],
    );
  }
}

class _DecimalDomain {
  final double min;
  final double max;
  const _DecimalDomain(this.min, this.max);
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
          (point.longitude <
              (xj - xi) * (point.latitude - yi) / (yj - yi) + xi);
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
        return LatLng(
            (coord[1] as num).toDouble(), (coord[0] as num).toDouble());
      }).toList();
      if (_pointInPolygon(point, pts)) return f;
    }
    return null;
  }

  // ── Score and color helpers ──────────────────────────────────────────────

  double _heatC(Map<String, dynamic> p) {
    return (p['temperature_c'] as num?)?.toDouble() ??
        (p['lst_pred_celsius'] as num?)?.toDouble() ??
        22.0 +
            (((p['heat_stress_score'] as num?)?.toDouble() ?? 0) / 100) * 12.0;
  }

  double _flood(Map<String, dynamic> p) {
    return (p['flood_risk_score'] as num?)?.toDouble() ?? 0;
  }

  double _norm(double value, double min, double max) {
    return ((value - min) / (max - min)).clamp(0.0, 1.0);
  }

  double _rawValue(Map<String, dynamic> p, [_MapLayer? layer]) {
    final heatT = _norm(_heatC(p), _heatMinC, _heatMaxC);
    final floodT = _norm(_flood(p), _floodMin, _floodMax);
    return switch (layer ?? widget.layer) {
      _MapLayer.heat => _heatC(p),
      _MapLayer.flood => _flood(p),
      _MapLayer.combined => (0.58 * heatT + 0.42 * floodT).clamp(0.0, 1.0),
    };
  }

  _DecimalDomain _domain() {
    final values = widget.features
        .map((f) => (f['properties'] as Map<String, dynamic>? ?? {}))
        .map(_rawValue)
        .where((v) => v.isFinite)
        .toList();
    if (values.isEmpty) {
      return switch (widget.layer) {
        _MapLayer.heat => const _DecimalDomain(_heatMinC, _heatMaxC),
        _MapLayer.flood => const _DecimalDomain(_floodMin, _floodMax),
        _MapLayer.combined => const _DecimalDomain(0.0, 1.0),
      };
    }
    var min = values.reduce((a, b) => a < b ? a : b);
    var max = values.reduce((a, b) => a > b ? a : b);
    final pad = switch (widget.layer) {
      _MapLayer.heat => 0.05,
      _MapLayer.flood => 0.01,
      _MapLayer.combined => 0.005,
    };
    if ((max - min).abs() < pad) {
      min -= pad;
      max += pad;
    }
    return _DecimalDomain(min, max);
  }

  double _intensity(Map<String, dynamic> p, _DecimalDomain domain) {
    return _norm(_rawValue(p), domain.min, domain.max);
  }

  Color _scoreColor(double intensity) {
    final t = intensity.clamp(0.0, 1.0);
    final palette = _layerPalette(widget.layer);
    return Color.lerp(palette.low, palette.high, t)!;
  }

  @override
  Widget build(BuildContext context) {
    final selected = ref.watch(_selectedWardProvider);
    final domain = _domain();
    final domainDigits = switch (widget.layer) {
      _MapLayer.heat => 2,
      _MapLayer.flood => 2,
      _MapLayer.combined => 3,
    };
    final domainSuffix = widget.layer == _MapLayer.heat ? '°C' : '';

    final polygons = widget.features
        .map((f) {
          final props = f['properties'] as Map<String, dynamic>? ?? {};
          final geom = f['geometry'] as Map<String, dynamic>? ?? {};
          if (geom['type'] != 'Polygon') return null;

          final rings = (geom['coordinates'] as List).first as List;
          final pts = rings.map((c) {
            final coord = c as List;
            return LatLng(
                (coord[1] as num).toDouble(), (coord[0] as num).toDouble());
          }).toList();

          final col = _scoreColor(_intensity(props, domain));
          final isSelected = selected != null &&
              (selected['properties']?['ward_id']) == props['ward_id'];
          final borderColor = Color.lerp(
            col,
            _layerPalette(widget.layer).high,
            0.55,
          )!;

          return Polygon(
            points: pts,
            color: col.withOpacity(isSelected ? 0.9 : 0.76),
            borderColor: isSelected ? Colors.white : borderColor,
            borderStrokeWidth: isSelected ? 2.4 : 1.15,
            isFilled: true,
          );
        })
        .whereType<Polygon>()
        .toList();

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
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.62),
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
              const SizedBox(height: 6),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.62),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  '${domain.min.toStringAsFixed(domainDigits)}$domainSuffix - ${domain.max.toStringAsFixed(domainDigits)}$domainSuffix',
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 10,
                      fontWeight: FontWeight.w700),
                ),
              ),
            ],
          ),
        ),
        Positioned(
          left: 12,
          right: 12,
          bottom: selected == null ? 16 : 150,
          child: IgnorePointer(
            child: DecoratedBox(
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.55),
                borderRadius: BorderRadius.circular(999),
              ),
              child: Padding(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
                child: Text(
                  'Decimal color stretch: ${domain.min.toStringAsFixed(domainDigits)}$domainSuffix to ${domain.max.toStringAsFixed(domainDigits)}$domainSuffix',
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 11,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ),
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
    final temp = (props['temperature_c'] as num?)?.toDouble();
    final lst = (props['lst_pred_celsius'] as num?)?.toDouble();
    final displayTemp = temp ?? lst;
    final riskLevel = props['risk_level'] as String? ?? 'unknown';

    final riskColor = switch (riskLevel) {
      'critical' => const Color(0xFFD32F2F),
      'high' => const Color(0xFFF57C00),
      'medium' => const Color(0xFFFBC02D),
      _ => const Color(0xFF388E3C),
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
                  padding:
                      const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
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
                if (displayTemp != null)
                  _InfoChip(
                      label:
                          '${temp != null ? 'Air' : 'LST'} ${displayTemp.toStringAsFixed(2)}°C'),
                _InfoChip(label: 'Flood ${flood.toStringAsFixed(2)}'),
                if (props['ndvi'] != null)
                  _InfoChip(
                      label:
                          'NDVI ${(props['ndvi'] as num).toStringAsFixed(2)}'),
                const Spacer(),
                TextButton.icon(
                  icon: const Icon(Icons.open_in_new, size: 14),
                  label: const Text('Details', style: TextStyle(fontSize: 12)),
                  style: TextButton.styleFrom(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
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
                    fontSize: 11, color: color, fontWeight: FontWeight.w600)),
            const Spacer(),
            Text('${value.toStringAsFixed(0)}/100',
                style: TextStyle(
                    fontSize: 11, fontWeight: FontWeight.w700, color: color)),
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
