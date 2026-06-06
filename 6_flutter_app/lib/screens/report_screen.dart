import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:image_picker/image_picker.dart';

import '../models/ward.dart';
import '../services/base_url.dart';
import '../widgets/app_theme.dart';

// ────────────────────────────────────────────────────────────────────────────
// Report types
// ────────────────────────────────────────────────────────────────────────────

const _reportTypes = [
  _ReportType('flood', 'Flooding', Icons.water, Colors.blue),
  _ReportType('heat_wave', 'Heat Wave', Icons.thermostat, Color(0xFFD32F2F)),
  _ReportType('waterlogging', 'Waterlogging', Icons.flood, Color(0xFF1565C0)),
  _ReportType('air_quality', 'Air Quality', Icons.air, Colors.blueGrey),
  _ReportType('tree_fall', 'Tree Fall', Icons.park, Colors.green),
  _ReportType('other', 'Other', Icons.report_problem, Colors.orange),
];

// ────────────────────────────────────────────────────────────────────────────
// State
// ────────────────────────────────────────────────────────────────────────────

enum _SubmitState { idle, loading, success, error }

final _submitStateProvider =
    StateProvider<_SubmitState>((_) => _SubmitState.idle);
final _submitErrorProvider = StateProvider<String>((_) => '');
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
  final _formKey = GlobalKey<FormState>();
  final _descCtrl = TextEditingController();
  final _wardCtrl = TextEditingController();
  final _picker = ImagePicker();

  int? _wardId;
  String? _autoWardName;
  bool _wardResolving = false;
  String? _wardResolveError;
  String _reportType = 'flood';
  int _severity = 3;
  double _lat = 12.9716;
  double _lon = 77.5946;

  // Image state
  Uint8List? _imageBytes;
  String? _imageFileName;

  @override
  void initState() {
    super.initState();
    _resolveWardFromCoordinates();
  }

  @override
  void dispose() {
    _descCtrl.dispose();
    _wardCtrl.dispose();
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
      _imageBytes = bytes;
      _imageFileName = picked.name;
    });
  }

  void _removeImage() => setState(() {
        _imageBytes = null;
      _imageFileName = null;
      });

  Future<void> _resolveWardFromCoordinates() async {
    setState(() {
      _wardResolving = true;
      _wardResolveError = null;
    });

    try {
      final dio = Dio(BaseOptions(
        baseUrl: apiBaseUrl,
        connectTimeout: const Duration(seconds: 10),
        receiveTimeout: const Duration(seconds: 20),
      ));
      final resp = await dio.get('/reports/resolve-ward', queryParameters: {
        'latitude': _lat,
        'longitude': _lon,
      });
      final data = resp.data as Map<String, dynamic>;
      final wardId = (data['ward_id'] as num).toInt();
      final wardName = data['ward_name']?.toString() ?? 'Ward $wardId';
      if (!mounted) return;
      setState(() {
        _wardId = wardId;
        _autoWardName = wardName;
        _wardCtrl.text = wardName;
        _wardResolving = false;
      });
    } on DioException catch (e) {
      if (!mounted) return;
      setState(() {
        _wardResolving = false;
        _wardResolveError =
            e.response?.data?['detail']?.toString() ?? e.message;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _wardResolving = false;
        _wardResolveError = e.toString();
      });
    }
  }

  // ── Submit ────────────────────────────────────────────────────────────────
  Future<void> _submit() async {
    if (_wardId == null && !_wardResolving) {
      await _resolveWardFromCoordinates();
    }
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
          'ward_id': _wardId,
          'latitude': _lat,
          'longitude': _lon,
          'report_type': _reportType,
          'severity': _severity,
          'description':
              _descCtrl.text.trim().isEmpty ? '' : _descCtrl.text.trim(),
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
          'ward_id': _wardId,
          'latitude': _lat,
          'longitude': _lon,
          'report_type': _reportType,
          'severity': _severity,
          'description':
              _descCtrl.text.trim().isEmpty ? null : _descCtrl.text.trim(),
        });
        responseData = resp.data as Map<String, dynamic>;
      }

      ref.read(_submitResultProvider.notifier).state = responseData;
      ref.read(_submitStateProvider.notifier).state = _SubmitState.success;
      _formKey.currentState!.reset();
      _descCtrl.clear();
      setState(() {
        _reportType = 'flood';
        _severity = 3;
        _imageBytes = null;
        _imageFileName = null;
      });
    } on DioException catch (e) {
      ref.read(_submitErrorProvider.notifier).state =
          e.response?.data?['detail']?.toString() ??
              e.message ??
              'Unknown error';
      ref.read(_submitStateProvider.notifier).state = _SubmitState.error;
    } catch (e) {
      ref.read(_submitErrorProvider.notifier).state = e.toString();
      ref.read(_submitStateProvider.notifier).state = _SubmitState.error;
    }
  }

  // ── Build ─────────────────────────────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    final submitState = ref.watch(_submitStateProvider);
    final submitError = ref.watch(_submitErrorProvider);
    final submitResult = ref.watch(_submitResultProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Report Incident'),
        actions: const [ThemeModeMenu()],
      ),
      body: NatureBackdrop(
        child: SingleChildScrollView(
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
                      ref.read(_submitStateProvider.notifier).state =
                          _SubmitState.idle;
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
                        const Icon(Icons.error_outline,
                            color: Colors.red, size: 22),
                        const SizedBox(width: 10),
                        Expanded(
                            child: Text('Failed: $submitError',
                                style: const TextStyle(color: Colors.red))),
                        TextButton(
                          onPressed: () => ref
                              .read(_submitStateProvider.notifier)
                              .state = _SubmitState.idle,
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
                          width: 44,
                          height: 44,
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
                              Text('Citizen Report',
                                  style: TextStyle(
                                      fontWeight: FontWeight.w700,
                                      fontSize: 16)),
                              SizedBox(height: 2),
                              Text(
                                  'Upload a photo — AI will detect flooding automatically',
                                  style: TextStyle(
                                      fontSize: 12, color: Colors.grey)),
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
                  fileName: _imageFileName,
                  onPick: _pickImage,
                  onRemove: _removeImage,
                ),
                const SizedBox(height: 16),

                // ── Auto ward from coordinates ──────────────────────────────
                _AutoWardField(
                  controller: _wardCtrl,
                  wardId: _wardId,
                  wardName: _autoWardName,
                  resolving: _wardResolving,
                  error: _wardResolveError,
                  onRefresh: _resolveWardFromCoordinates,
                ),
                const SizedBox(height: 16),

                // ── Incident type ────────────────────────────────────────────
                const Text('Incident Type',
                    style:
                        TextStyle(fontWeight: FontWeight.w600, fontSize: 14)),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: _reportTypes.map((rt) {
                    final sel = _reportType == rt.value;
                    return GestureDetector(
                      onTap: () => setState(() => _reportType = rt.value),
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 12, vertical: 8),
                        decoration: BoxDecoration(
                          color: sel
                              ? rt.color.withOpacity(0.15)
                              : Colors.grey.shade100,
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(
                              color: sel ? rt.color : Colors.grey.shade300,
                              width: sel ? 1.5 : 1),
                        ),
                        child: Row(mainAxisSize: MainAxisSize.min, children: [
                          Icon(rt.icon,
                              size: 16, color: sel ? rt.color : Colors.grey),
                          const SizedBox(width: 6),
                          Text(rt.label,
                              style: TextStyle(
                                fontSize: 13,
                                color: sel ? rt.color : Colors.grey,
                                fontWeight:
                                    sel ? FontWeight.w700 : FontWeight.normal,
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
                      style:
                          TextStyle(fontWeight: FontWeight.w600, fontSize: 14)),
                  const Spacer(),
                  _SeverityLabel(severity: _severity),
                ]),
                Slider(
                  value: _severity.toDouble(),
                  min: 1,
                  max: 5,
                  divisions: 4,
                  label: _severityLabel(_severity),
                  activeColor: _severityColor(_severity),
                  onChanged: (v) => setState(() => _severity = v.toInt()),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 8),
                  child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: const [
                        Text('Minor',
                            style: TextStyle(fontSize: 10, color: Colors.grey)),
                        Text('Extreme',
                            style: TextStyle(fontSize: 10, color: Colors.grey)),
                      ]),
                ),
                const SizedBox(height: 16),

                // ── Description ──────────────────────────────────────────────
                TextFormField(
                  controller: _descCtrl,
                  maxLines: 3,
                  maxLength: 500,
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
                    color: Colors.blue.shade50,
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: Colors.blue.shade200),
                  ),
                  child: Row(children: [
                    Icon(Icons.my_location,
                        size: 16, color: Colors.blue.shade700),
                    const SizedBox(width: 8),
                    Text(
                        'Location: ${_lat.toStringAsFixed(4)}, ${_lon.toStringAsFixed(4)}',
                        style: TextStyle(
                            fontSize: 12, color: Colors.blue.shade700)),
                  ]),
                ),
                const SizedBox(height: 24),

                // ── Submit ───────────────────────────────────────────────────
                SizedBox(
                  height: 52,
                  child: ElevatedButton.icon(
                    icon: submitState == _SubmitState.loading
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(
                                strokeWidth: 2, color: Colors.white))
                        : Icon(_imageBytes != null
                            ? Icons.auto_awesome
                            : Icons.send),
                    label: Text(
                      submitState == _SubmitState.loading
                          ? (_imageBytes != null
                              ? 'Analysing image…'
                              : 'Submitting…')
                          : (_imageBytes != null
                              ? 'Submit + Analyse Photo'
                              : 'Submit Report'),
                      style: const TextStyle(fontSize: 15),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF1B6B3A),
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12)),
                    ),
                    onPressed:
                        submitState == _SubmitState.loading ? null : _submit,
                  ),
                ),
                const SizedBox(height: 24),
              ],
            ),
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
  final String? fileName;
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
            child: Image.memory(imageBytes!,
                height: 200, width: double.infinity, fit: BoxFit.cover),
          ),
          Positioned(
            top: 8,
            right: 8,
            child: GestureDetector(
              onTap: onRemove,
              child: Container(
                padding: const EdgeInsets.all(6),
                decoration: BoxDecoration(
                    color: Colors.black54,
                    borderRadius: BorderRadius.circular(20)),
                child: const Icon(Icons.close, color: Colors.white, size: 16),
              ),
            ),
          ),
          Positioned(
            bottom: 8,
            left: 8,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                  color: Colors.black54,
                  borderRadius: BorderRadius.circular(6)),
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
          border: Border.all(
              color: Colors.blue.shade200,
              width: 1.5,
              style: BorderStyle.solid),
        ),
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
          Icon(Icons.add_a_photo, size: 36, color: Colors.blue.shade400),
          const SizedBox(height: 8),
          Text('Tap to upload photo',
              style: TextStyle(
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
// Coordinate-derived ward display
// ────────────────────────────────────────────────────────────────────────────

class _AutoWardField extends StatelessWidget {
  final TextEditingController controller;
  final int? wardId;
  final String? wardName;
  final bool resolving;
  final String? error;
  final VoidCallback onRefresh;

  const _AutoWardField({
    required this.controller,
    required this.wardId,
    required this.wardName,
    required this.resolving,
    required this.error,
    required this.onRefresh,
  });

  @override
  Widget build(BuildContext context) {
    final label = wardId == null
        ? 'Resolving ward from coordinates...'
        : '${wardName ?? 'Ward $wardId'} · Ward $wardId';

    return TextFormField(
      controller: controller,
      readOnly: true,
      decoration: InputDecoration(
        labelText: 'Ward auto-selected from GPS',
        prefixIcon: resolving
            ? const Padding(
                padding: EdgeInsets.all(12),
                child: SizedBox(
                  width: 18,
                  height: 18,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
              )
            : const Icon(Icons.my_location),
        suffixIcon: IconButton(
          tooltip: 'Refresh ward from coordinates',
          icon: const Icon(Icons.refresh),
          onPressed: resolving ? null : onRefresh,
        ),
        border: const OutlineInputBorder(),
        hintText: label,
        helperText: error == null
            ? 'Citizens cannot manually change ward; it is derived from coordinates.'
            : null,
        errorText: error,
      ),
      validator: (_) {
        if (wardId == null) {
          return 'Wait for ward to be selected from location';
        }
        return null;
      },
    );
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Searchable ward selector
// ────────────────────────────────────────────────────────────────────────────

class _WardAutocompleteField extends StatefulWidget {
  final List<WardRisk> wards;
  final TextEditingController controller;
  final FocusNode focusNode;
  final int? selectedWardId;
  final ValueChanged<int?> onChanged;

  const _WardAutocompleteField({
    required this.wards,
    required this.controller,
    required this.focusNode,
    required this.selectedWardId,
    required this.onChanged,
  });

  @override
  State<_WardAutocompleteField> createState() => _WardAutocompleteFieldState();
}

class _WardAutocompleteFieldState extends State<_WardAutocompleteField> {
  bool _showOptions = false;

  static String _label(WardRisk ward) => ward.displayName;

  @override
  void initState() {
    super.initState();
    widget.focusNode.addListener(_handleFocusChange);
  }

  @override
  void didUpdateWidget(covariant _WardAutocompleteField oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.focusNode != widget.focusNode) {
      oldWidget.focusNode.removeListener(_handleFocusChange);
      widget.focusNode.addListener(_handleFocusChange);
    }
  }

  @override
  void dispose() {
    widget.focusNode.removeListener(_handleFocusChange);
    super.dispose();
  }

  void _handleFocusChange() {
    if (!mounted) return;
    if (widget.focusNode.hasFocus) {
      setState(() => _showOptions = true);
    }
  }

  int? _resolveWardId(String input) {
    final query = input.trim().toLowerCase();
    if (query.isEmpty) return null;

    for (final ward in widget.wards) {
      final name = ward.displayName.toLowerCase();
      if (query == name) {
        return ward.wardId;
      }
    }
    return null;
  }

  Iterable<WardRisk> _optionsFor(String input) {
    final query = input.trim().toLowerCase();
    final sorted = [...widget.wards]
      ..sort((a, b) => a.wardId.compareTo(b.wardId));

    if (query.isEmpty) return sorted.take(12);

    final matches = sorted.where((ward) {
      final name = ward.displayName.toLowerCase();
      return name.contains(query);
    }).toList();

    matches.sort((a, b) {
      final aName = a.displayName.toLowerCase();
      final bName = b.displayName.toLowerCase();
      final aExact = aName == query;
      final bExact = bName == query;
      if (aExact != bExact) return aExact ? -1 : 1;
      final aStarts = aName.startsWith(query);
      final bStarts = bName.startsWith(query);
      if (aStarts != bStarts) return aStarts ? -1 : 1;
      return a.wardId.compareTo(b.wardId);
    });

    return matches.take(20);
  }

  void _selectWard(WardRisk ward) {
    widget.controller.value = TextEditingValue(
      text: _label(ward),
      selection: TextSelection.collapsed(offset: _label(ward).length),
    );
    widget.onChanged(ward.wardId);
    setState(() => _showOptions = false);
    widget.focusNode.unfocus();
  }

  @override
  Widget build(BuildContext context) {
    final options = _optionsFor(widget.controller.text).toList();

    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      TextFormField(
        controller: widget.controller,
        focusNode: widget.focusNode,
        textInputAction: TextInputAction.search,
        decoration: InputDecoration(
          labelText: 'Ward',
          prefixIcon: const Icon(Icons.location_city),
          suffixIcon: widget.selectedWardId == null
              ? null
              : IconButton(
                  tooltip: 'Clear ward',
                  icon: const Icon(Icons.close, size: 18),
                  onPressed: () {
                    widget.controller.clear();
                    widget.onChanged(null);
                    setState(() => _showOptions = true);
                  },
                ),
          border: const OutlineInputBorder(),
          hintText: 'Search ward name',
        ),
        onTap: () => setState(() => _showOptions = true),
        onChanged: (value) {
          widget.onChanged(_resolveWardId(value));
          setState(() => _showOptions = true);
        },
        validator: (value) {
          final wardId = _resolveWardId(value ?? '');
          if (wardId == null) {
            return 'Select a ward name from the list';
          }
          return null;
        },
      ),
      if (_showOptions) ...[
        const SizedBox(height: 6),
        Container(
          constraints: const BoxConstraints(maxHeight: 260),
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surface,
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: Colors.grey.shade300),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.08),
                blurRadius: 12,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: options.isEmpty
              ? const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                  child: Text(
                    'No matching ward name',
                    style: TextStyle(color: Colors.grey, fontSize: 13),
                  ),
                )
              : ListView.separated(
                  padding: EdgeInsets.zero,
                  shrinkWrap: true,
                  itemCount: options.length,
                  separatorBuilder: (_, __) => Divider(
                    height: 1,
                    color: Colors.grey.shade200,
                  ),
                  itemBuilder: (context, index) {
                    final ward = options[index];
                    return GestureDetector(
                      behavior: HitTestBehavior.opaque,
                      onTapDown: (_) => _selectWard(ward),
                      child: Padding(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 14,
                          vertical: 10,
                        ),
                        child: Row(children: [
                          Container(
                            width: 34,
                            height: 34,
                            alignment: Alignment.center,
                            decoration: BoxDecoration(
                              color: const Color(0xFF1B6B3A)
                                  .withValues(alpha: 0.1),
                              shape: BoxShape.circle,
                            ),
                            child: const Icon(
                              Icons.location_city,
                              size: 17,
                              color: Color(0xFF1B6B3A),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Text(
                              ward.displayName,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),
                        ]),
                      ),
                    );
                  },
                ),
        ),
      ],
    ]);
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Success banner (shows AI result)
// ────────────────────────────────────────────────────────────────────────────

class _SuccessBanner extends StatelessWidget {
  final Map<String, dynamic> result;
  final VoidCallback onDismiss;

  const _SuccessBanner({required this.result, required this.onDismiss});

  String _aiTitle(String? label, bool floodPredicted) {
    if (floodPredicted) return 'Flooding Detected';
    switch (label) {
      case 'poor_air_quality':
        return 'Poor Air Quality Detected';
      case 'waterlogging':
        return 'Waterlogging Detected';
      case 'flood':
        return 'Flooding Detected';
      case 'heat_wave_dry_conditions':
        return 'Heat/Dry Conditions Detected';
      case 'tree_fall_detected':
        return 'Tree Fall Detected';
      case 'no_visible_air_pollution':
        return 'No Visible Smog Detected';
      case 'no_visible_waterlogging':
        return 'No Visible Waterlogging';
      case 'no_visible_flood':
        return 'No Visible Flooding';
      case 'no_visible_heat_wave':
        return 'No Visible Heat Damage';
      case 'no_visible_tree_fall':
        return 'No Visible Tree Fall';
      default:
        return 'Report Submitted';
    }
  }

  IconData _aiIcon(String? label, bool floodPredicted) {
    if (floodPredicted || label == 'flood' || label == 'waterlogging') {
      return Icons.warning_amber;
    }
    if (label == 'poor_air_quality') return Icons.air;
    if (label == 'heat_wave_dry_conditions') return Icons.thermostat;
    if (label == 'tree_fall_detected') return Icons.park;
    return Icons.check_circle;
  }

  bool _isWarningResult(String? label, bool floodPredicted) {
    return floodPredicted ||
        label == 'poor_air_quality' ||
        label == 'waterlogging' ||
        label == 'flood' ||
        label == 'heat_wave_dry_conditions' ||
        label == 'tree_fall_detected';
  }

  @override
  Widget build(BuildContext context) {
    final floodPredicted = result['flood_predicted'] == true;
    final aiLabel = result['ai_label']?.toString();
    final aiConfidence =
        ((result['ai_confidence'] ?? result['flood_confidence']) as num?)
            ?.toDouble();
    final alertSent = result['alert_sent'] == true;
    final warning = _isWarningResult(aiLabel, floodPredicted);

    return Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
      // Main result card
      Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: warning ? Colors.orange.shade50 : Colors.green.shade50,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
              color: warning ? Colors.orange.shade300 : Colors.green.shade300),
        ),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Icon(_aiIcon(aiLabel, floodPredicted),
                color: warning ? Colors.orange.shade800 : Colors.green,
                size: 22),
            const SizedBox(width: 10),
            Expanded(
                child: Text(
              _aiTitle(aiLabel, floodPredicted),
              style: TextStyle(
                fontWeight: FontWeight.w800,
                fontSize: 15,
                color: warning ? Colors.orange.shade900 : Colors.green.shade800,
              ),
            )),
            IconButton(
                onPressed: onDismiss,
                icon: const Icon(Icons.close, size: 18),
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints()),
          ]),
          if (aiConfidence != null) ...[
            const SizedBox(height: 10),
            Text('AI Confidence: ${(aiConfidence * 100).toStringAsFixed(0)}%',
                style: TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 13,
                    color: warning
                        ? Colors.orange.shade800
                        : Colors.green.shade700)),
            const SizedBox(height: 6),
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: aiConfidence.clamp(0.0, 1.0).toDouble(),
                minHeight: 8,
                backgroundColor: Colors.grey.shade200,
                valueColor: AlwaysStoppedAnimation<Color>(
                    warning ? Colors.orange : Colors.green),
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
                  color: Colors.red.shade100,
                  borderRadius: BorderRadius.circular(6)),
              child: Row(mainAxisSize: MainAxisSize.min, children: const [
                Icon(Icons.notifications_active, size: 14, color: Colors.red),
                SizedBox(width: 6),
                Text('Alert sent to all citizens + BBMP',
                    style: TextStyle(
                        fontSize: 12,
                        color: Colors.red,
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
      Colors.green,
      Colors.green,
      const Color(0xFF8BC34A),
      const Color(0xFFFBC02D),
      const Color(0xFFF57C00),
      const Color(0xFFD32F2F),
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
        color: color.withOpacity(0.12),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.4)),
      ),
      child: Text('$severity — ${_severityLabel(severity)}',
          style: TextStyle(
              color: color, fontWeight: FontWeight.w700, fontSize: 12)),
    );
  }
}
