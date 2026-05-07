import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';

import '../services/base_url.dart';

// ────────────────────────────────────────────────────────────────────────────
// Report types
// ────────────────────────────────────────────────────────────────────────────

const _reportTypes = [
  _ReportType('flood',        'Flooding',     Icons.water,          Colors.blue),
  _ReportType('heat_wave',    'Heat Wave',    Icons.thermostat,     Color(0xFFD32F2F)),
  _ReportType('waterlogging', 'Waterlogging', Icons.flood,          Color(0xFF1565C0)),
  _ReportType('air_quality',  'Air Quality',  Icons.air,            Colors.blueGrey),
  _ReportType('tree_fall',    'Tree Fall',    Icons.park,           Colors.green),
  _ReportType('other',        'Other',        Icons.report_problem, Colors.orange),
];

// ────────────────────────────────────────────────────────────────────────────
// State
// ────────────────────────────────────────────────────────────────────────────

enum _SubmitState { idle, loading, success, error }

final _submitStateProvider = StateProvider<_SubmitState>((_) => _SubmitState.idle);
final _submitErrorProvider  = StateProvider<String>((_) => '');
final _submitResultProvider = StateProvider<Map<String, dynamic>?>((_) => null);

// ────────────────────────────────────────────────────────────────────────────
// Screen
// ────────────────────────────────────────────────────────────────────────────

class ReportScreen extends ConsumerStatefulWidget {
  const ReportScreen({super.key});

  @override
  ConsumerState<ReportScreen> createState() => _ReportScreenState();
}

class _ReportScreenState extends ConsumerState<ReportScreen> {
  final _formKey    = GlobalKey<FormState>();
  final _descCtrl   = TextEditingController();
  final _picker     = ImagePicker();

  int?          _wardId;
  String        _reportType = 'flood';
  int           _severity   = 3;
  double        _lat        = 12.9716;
  double        _lon        = 77.5946;

  // Image state
  Uint8List?    _imageBytes;
  String?       _imageFileName;

  @override
  void dispose() {
    _descCtrl.dispose();
    super.dispose();
  }

  // ── Pick image ────────────────────────────────────────────────────────────
  Future<void> _pickImage() async {
    final picked = await _picker.pickImage(
      source: ImageSource.gallery,
      maxWidth: 1920,
      maxHeight: 1920,
      imageQuality: 85,
    );
    if (picked == null) return;
    final bytes = await picked.readAsBytes();
    setState(() {
      _imageBytes    = bytes;
      _imageFileName = picked.name;
    });
  }

  void _removeImage() => setState(() { _imageBytes = null; _imageFileName = null; });

  // ── Submit ────────────────────────────────────────────────────────────────
  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    ref.read(_submitStateProvider.notifier).state = _SubmitState.loading;
    ref.read(_submitResultProvider.notifier).state = null;

    try {
      final dio = Dio(BaseOptions(
        baseUrl: apiBaseUrl,
        connectTimeout: const Duration(seconds: 15),
        receiveTimeout: const Duration(seconds: 30),
      ));

      late Map<String, dynamic> responseData;

      if (_imageBytes != null) {
        // Multipart POST with image → AI flood classification
        final formData = FormData.fromMap({
          'ward_id':     _wardId,
          'latitude':    _lat,
          'longitude':   _lon,
          'report_type': _reportType,
          'severity':    _severity,
          'description': _descCtrl.text.trim().isEmpty ? '' : _descCtrl.text.trim(),
          'image': MultipartFile.fromBytes(
            _imageBytes!,
            filename: _imageFileName ?? 'report.jpg',
          ),
        });
        final resp = await dio.post('/reports/with-image', data: formData);
        responseData = resp.data as Map<String, dynamic>;
      } else {
        // JSON POST without image
        final resp = await dio.post('/reports/', data: {
          'ward_id':     _wardId,
          'latitude':    _lat,
          'longitude':   _lon,
          'report_type': _reportType,
          'severity':    _severity,
          'description': _descCtrl.text.trim().isEmpty ? null : _descCtrl.text.trim(),
        });
        responseData = resp.data as Map<String, dynamic>;
      }

      ref.read(_submitResultProvider.notifier).state = responseData;
      ref.read(_submitStateProvider.notifier).state  = _SubmitState.success;
      _formKey.currentState!.reset();
      _descCtrl.clear();
      setState(() {
        _wardId      = null;
        _reportType  = 'flood';
        _severity    = 3;
        _imageBytes  = null;
        _imageFileName = null;
      });
    } on DioException catch (e) {
      ref.read(_submitErrorProvider.notifier).state =
          e.response?.data?['detail']?.toString() ?? e.message ?? 'Unknown error';
      ref.read(_submitStateProvider.notifier).state = _SubmitState.error;
    } catch (e) {
      ref.read(_submitErrorProvider.notifier).state = e.toString();
      ref.read(_submitStateProvider.notifier).state = _SubmitState.error;
    }
  }

  // ── Build ─────────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    final submitState  = ref.watch(_submitStateProvider);
    final submitError  = ref.watch(_submitErrorProvider);
    final submitResult = ref.watch(_submitResultProvider);

    return Scaffold(
      appBar: AppBar(title: const Text('Report Incident')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [

              // ── Success banner ──────────────────────────────────────────
              if (submitState == _SubmitState.success && submitResult != null)
                _SuccessBanner(
                  result: submitResult,
                  onDismiss: () {
                    ref.read(_submitStateProvider.notifier).state = _SubmitState.idle;
                    ref.read(_submitResultProvider.notifier).state = null;
                  },
                ),

              // ── Error banner ────────────────────────────────────────────
              if (submitState == _SubmitState.error) ...[
                Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: Colors.red.shade50,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.red.shade300),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.error_outline, color: Colors.red, size: 22),
                      const SizedBox(width: 10),
                      Expanded(child: Text('Failed: $submitError',
                          style: const TextStyle(color: Colors.red))),
                      TextButton(
                        onPressed: () => ref.read(_submitStateProvider.notifier).state = _SubmitState.idle,
                        child: const Text('Retry'),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
              ],

              // ── Header ──────────────────────────────────────────────────
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(14),
                  child: Row(
                    children: [
                      Container(
                        width: 44, height: 44,
                        decoration: BoxDecoration(
                          color: const Color(0xFF1B6B3A).withOpacity(0.12),
                          shape: BoxShape.circle,
                        ),
                        child: const Icon(Icons.report_gmailerrorred,
                            color: Color(0xFF1B6B3A), size: 24),
                      ),
                      const SizedBox(width: 14),
                      const Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('Citizen Report', style: TextStyle(
                                fontWeight: FontWeight.w700, fontSize: 16)),
                            SizedBox(height: 2),
                            Text('Upload a photo — AI will detect flooding automatically',
                                style: TextStyle(fontSize: 12, color: Colors.grey)),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),

              // ── Image Upload ─────────────────────────────────────────────
              _ImageUploadSection(
                imageBytes: _imageBytes,
                fileName:   _imageFileName,
                onPick:     _pickImage,
                onRemove:   _removeImage,
              ),
              const SizedBox(height: 16),

              // ── Ward ID ──────────────────────────────────────────────────
              TextFormField(
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: 'Ward ID (1–198)',
                  prefixIcon: Icon(Icons.location_city),
                  border: OutlineInputBorder(),
                  hintText: 'e.g. 42',
                ),
                onChanged: (v) => setState(() => _wardId = int.tryParse(v)),
                validator: (v) {
                  final n = int.tryParse(v ?? '');
                  if (n == null || n < 1 || n > 198) return 'Enter a valid ward ID (1–198)';
                  return null;
                },
              ),
              const SizedBox(height: 16),

              // ── Incident type ────────────────────────────────────────────
              const Text('Incident Type',
                  style: TextStyle(fontWeight: FontWeight.w600, fontSize: 14)),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8, runSpacing: 8,
                children: _reportTypes.map((rt) {
                  final sel = _reportType == rt.value;
                  return GestureDetector(
                    onTap: () => setState(() => _reportType = rt.value),
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      decoration: BoxDecoration(
                        color: sel ? rt.color.withOpacity(0.15) : Colors.grey.shade100,
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(
                            color: sel ? rt.color : Colors.grey.shade300,
                            width: sel ? 1.5 : 1),
                      ),
                      child: Row(mainAxisSize: MainAxisSize.min, children: [
                        Icon(rt.icon, size: 16, color: sel ? rt.color : Colors.grey),
                        const SizedBox(width: 6),
                        Text(rt.label, style: TextStyle(
                          fontSize: 13,
                          color: sel ? rt.color : Colors.grey,
                          fontWeight: sel ? FontWeight.w700 : FontWeight.normal,
                        )),
                      ]),
                    ),
                  );
                }).toList(),
              ),
              const SizedBox(height: 16),

              // ── Severity ─────────────────────────────────────────────────
              Row(children: [
                const Text('Severity',
                    style: TextStyle(fontWeight: FontWeight.w600, fontSize: 14)),
                const Spacer(),
                _SeverityLabel(severity: _severity),
              ]),
              Slider(
                value: _severity.toDouble(), min: 1, max: 5, divisions: 4,
                label: _severityLabel(_severity),
                activeColor: _severityColor(_severity),
                onChanged: (v) => setState(() => _severity = v.toInt()),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8),
                child: Row(mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: const [
                  Text('Minor',   style: TextStyle(fontSize: 10, color: Colors.grey)),
                  Text('Extreme', style: TextStyle(fontSize: 10, color: Colors.grey)),
                ]),
              ),
              const SizedBox(height: 16),

              // ── Description ──────────────────────────────────────────────
              TextFormField(
                controller: _descCtrl,
                maxLines: 3, maxLength: 500,
                decoration: const InputDecoration(
                  labelText: 'Description (optional)',
                  prefixIcon: Padding(
                    padding: EdgeInsets.only(bottom: 48),
                    child: Icon(Icons.notes),
                  ),
                  border: OutlineInputBorder(),
                  hintText: 'Describe what you see…',
                  alignLabelWithHint: true,
                ),
              ),
              const SizedBox(height: 16),

              // ── Location ─────────────────────────────────────────────────
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.blue.shade50, borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: Colors.blue.shade200),
                ),
                child: Row(children: [
                  Icon(Icons.my_location, size: 16, color: Colors.blue.shade700),
                  const SizedBox(width: 8),
                  Text('Location: ${_lat.toStringAsFixed(4)}, ${_lon.toStringAsFixed(4)}',
                      style: TextStyle(fontSize: 12, color: Colors.blue.shade700)),
                ]),
              ),
              const SizedBox(height: 24),

              // ── Submit ───────────────────────────────────────────────────
              SizedBox(
                height: 52,
                child: ElevatedButton.icon(
                  icon: submitState == _SubmitState.loading
                      ? const SizedBox(width: 18, height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                      : Icon(_imageBytes != null ? Icons.auto_awesome : Icons.send),
                  label: Text(
                    submitState == _SubmitState.loading
                        ? (_imageBytes != null ? 'Analysing image…' : 'Submitting…')
                        : (_imageBytes != null ? 'Submit + Analyse Photo' : 'Submit Report'),
                    style: const TextStyle(fontSize: 15),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF1B6B3A),
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                  onPressed: submitState == _SubmitState.loading ? null : _submit,
                ),
              ),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Image upload section
// ────────────────────────────────────────────────────────────────────────────

class _ImageUploadSection extends StatelessWidget {
  final Uint8List? imageBytes;
  final String?    fileName;
  final VoidCallback onPick;
  final VoidCallback onRemove;

  const _ImageUploadSection({
    required this.imageBytes,
    required this.fileName,
    required this.onPick,
    required this.onRemove,
  });

  @override
  Widget build(BuildContext context) {
    if (imageBytes != null) {
      return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Stack(children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: Image.memory(imageBytes!, height: 200, width: double.infinity,
                fit: BoxFit.cover),
          ),
          Positioned(top: 8, right: 8,
            child: GestureDetector(
              onTap: onRemove,
              child: Container(
                padding: const EdgeInsets.all(6),
                decoration: BoxDecoration(
                    color: Colors.black54, borderRadius: BorderRadius.circular(20)),
                child: const Icon(Icons.close, color: Colors.white, size: 16),
              ),
            ),
          ),
          Positioned(bottom: 8, left: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                  color: Colors.black54, borderRadius: BorderRadius.circular(6)),
              child: Row(mainAxisSize: MainAxisSize.min, children: [
                const Icon(Icons.auto_awesome, color: Colors.white, size: 12),
                const SizedBox(width: 4),
                Text('AI flood detection will run on submit',
                    style: const TextStyle(color: Colors.white, fontSize: 11)),
              ]),
            ),
          ),
        ]),
      ]);
    }

    return GestureDetector(
      onTap: onPick,
      child: Container(
        height: 130,
        decoration: BoxDecoration(
          color: Colors.blue.shade50,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.blue.shade200, width: 1.5,
              style: BorderStyle.solid),
        ),
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
          Icon(Icons.add_a_photo, size: 36, color: Colors.blue.shade400),
          const SizedBox(height: 8),
          Text('Tap to upload photo', style: TextStyle(
              fontWeight: FontWeight.w600, color: Colors.blue.shade700)),
          const SizedBox(height: 4),
          Text('AI will automatically detect flooding',
              style: TextStyle(fontSize: 11, color: Colors.blue.shade400)),
        ]),
      ),
    );
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Success banner (shows AI result)
// ────────────────────────────────────────────────────────────────────────────

class _SuccessBanner extends StatelessWidget {
  final Map<String, dynamic> result;
  final VoidCallback onDismiss;

  const _SuccessBanner({required this.result, required this.onDismiss});

  @override
  Widget build(BuildContext context) {
    final floodPredicted  = result['flood_predicted'] == true;
    final floodConfidence = (result['flood_confidence'] as num?)?.toDouble();
    final alertSent       = result['alert_sent'] == true;

    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      // Main result card
      Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: floodPredicted ? Colors.red.shade50 : Colors.green.shade50,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
              color: floodPredicted ? Colors.red.shade300 : Colors.green.shade300),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Icon(floodPredicted ? Icons.warning_amber : Icons.check_circle,
                color: floodPredicted ? Colors.red : Colors.green, size: 22),
            const SizedBox(width: 10),
            Expanded(child: Text(
              floodPredicted ? '🌊 Flooding Detected!' : '✅ Report Submitted',
              style: TextStyle(
                fontWeight: FontWeight.w800, fontSize: 15,
                color: floodPredicted ? Colors.red.shade800 : Colors.green.shade800,
              ),
            )),
            IconButton(onPressed: onDismiss, icon: const Icon(Icons.close, size: 18),
                padding: EdgeInsets.zero, constraints: const BoxConstraints()),
          ]),
          if (floodConfidence != null) ...[
            const SizedBox(height: 10),
            Text('AI Confidence: ${(floodConfidence * 100).toStringAsFixed(0)}%',
                style: TextStyle(
                    fontWeight: FontWeight.w600, fontSize: 13,
                    color: floodPredicted ? Colors.red.shade700 : Colors.green.shade700)),
            const SizedBox(height: 6),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: floodConfidence, minHeight: 8,
                backgroundColor: Colors.grey.shade200,
                valueColor: AlwaysStoppedAnimation<Color>(
                    floodPredicted ? Colors.red : Colors.green),
              ),
            ),
          ],
          const SizedBox(height: 10),
          Text(result['message']?.toString() ?? 'Report recorded.',
              style: const TextStyle(fontSize: 13)),
          if (alertSent) ...[
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
              decoration: BoxDecoration(
                  color: Colors.red.shade100, borderRadius: BorderRadius.circular(6)),
              child: Row(mainAxisSize: MainAxisSize.min, children: const [
                Icon(Icons.notifications_active, size: 14, color: Colors.red),
                SizedBox(width: 6),
                Text('Alert sent to all citizens + BBMP',
                    style: TextStyle(fontSize: 12, color: Colors.red,
                        fontWeight: FontWeight.w600)),
              ]),
            ),
          ],
        ]),
      ),
      const SizedBox(height: 12),
    ]);
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

class _ReportType {
  final String value, label;
  final IconData icon;
  final Color color;
  const _ReportType(this.value, this.label, this.icon, this.color);
}

String _severityLabel(int s) =>
    ['', 'Minor', 'Low', 'Moderate', 'High', 'Extreme'][s.clamp(1, 5)];

Color _severityColor(int s) => [
  Colors.green, Colors.green, const Color(0xFF8BC34A),
  const Color(0xFFFBC02D), const Color(0xFFF57C00), const Color(0xFFD32F2F),
][s.clamp(0, 5)];

class _SeverityLabel extends StatelessWidget {
  final int severity;
  const _SeverityLabel({required this.severity});

  @override
  Widget build(BuildContext context) {
    final color = _severityColor(severity);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
      decoration: BoxDecoration(
        color: color.withOpacity(0.12), borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.4)),
      ),
      child: Text('$severity — ${_severityLabel(severity)}',
          style: TextStyle(color: color, fontWeight: FontWeight.w700, fontSize: 12)),
    );
  }
}
