import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:google_fonts/google_fonts.dart';

final themeModeProvider = StateProvider<ThemeMode>((_) => ThemeMode.system);

const ncgCanopy = Color(0xFF0F5F37);
const ncgLeaf = Color(0xFF1F8A4C);
const ncgMist = Color(0xFFF3F8F4);

ThemeData buildNcgTheme(Brightness brightness) {
  final isDark = brightness == Brightness.dark;
  final scheme = ColorScheme.fromSeed(
    seedColor: ncgCanopy,
    brightness: brightness,
  );
  final base = ThemeData(colorScheme: scheme, useMaterial3: true);
  final surface = isDark ? const Color(0xFF0D1711) : ncgMist;
  final card = isDark ? const Color(0xFF13231A) : Colors.white.withOpacity(.92);

  return base.copyWith(
    scaffoldBackgroundColor: surface,
    textTheme: GoogleFonts.nunitoTextTheme(base.textTheme).apply(
      bodyColor: scheme.onSurface,
      displayColor: scheme.onSurface,
    ),
    appBarTheme: AppBarTheme(
      elevation: 0,
      centerTitle: false,
      backgroundColor: isDark ? const Color(0xFF102419) : ncgCanopy,
      foregroundColor: Colors.white,
      titleTextStyle: GoogleFonts.nunito(
        color: Colors.white,
        fontSize: 20,
        fontWeight: FontWeight.w900,
        letterSpacing: -0.2,
      ),
    ),
    cardTheme: base.cardTheme.copyWith(
      elevation: 0,
      color: card,
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(22),
        side: BorderSide(
          color: isDark
              ? Colors.white.withOpacity(.08)
              : ncgCanopy.withOpacity(.08),
        ),
      ),
    ),
    navigationBarTheme: NavigationBarThemeData(
      height: 72,
      backgroundColor:
          isDark ? const Color(0xFF102419) : Colors.white.withOpacity(.92),
      indicatorColor:
          isDark ? ncgLeaf.withOpacity(.34) : ncgCanopy.withOpacity(.12),
      labelTextStyle: MaterialStateProperty.resolveWith((states) {
        final selected = states.contains(MaterialState.selected);
        return GoogleFonts.nunito(
          fontSize: 11,
          fontWeight: selected ? FontWeight.w900 : FontWeight.w700,
          color: selected ? scheme.primary : scheme.onSurfaceVariant,
        );
      }),
      iconTheme: MaterialStateProperty.resolveWith((states) {
        final selected = states.contains(MaterialState.selected);
        return IconThemeData(
          color: selected ? scheme.primary : scheme.onSurfaceVariant,
          size: selected ? 25 : 23,
        );
      }),
    ),
    chipTheme: base.chipTheme.copyWith(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(999)),
      side: BorderSide(color: scheme.outlineVariant),
      labelStyle: GoogleFonts.nunito(fontWeight: FontWeight.w800),
    ),
  );
}

class ThemeModeMenu extends ConsumerWidget {
  const ThemeModeMenu({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final mode = ref.watch(themeModeProvider);
    final icon = switch (mode) {
      ThemeMode.dark => Icons.dark_mode,
      ThemeMode.light => Icons.light_mode,
      ThemeMode.system => Icons.brightness_auto,
    };

    return PopupMenuButton<ThemeMode>(
      tooltip: 'Theme mode',
      icon: Icon(icon),
      initialValue: mode,
      onSelected: (value) => ref.read(themeModeProvider.notifier).state = value,
      itemBuilder: (context) => const [
        PopupMenuItem(
          value: ThemeMode.light,
          child:
              ListTile(leading: Icon(Icons.light_mode), title: Text('Light')),
        ),
        PopupMenuItem(
          value: ThemeMode.dark,
          child: ListTile(leading: Icon(Icons.dark_mode), title: Text('Dark')),
        ),
        PopupMenuItem(
          value: ThemeMode.system,
          child: ListTile(
              leading: Icon(Icons.brightness_auto), title: Text('System')),
        ),
      ],
    );
  }
}

class NcgMark extends StatelessWidget {
  final double size;
  const NcgMark({super.key, this.size = 42});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: Theme.of(context).brightness == Brightness.dark
            ? const Color(0xFF142C1E)
            : Colors.white,
        borderRadius: BorderRadius.circular(size * .32),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(.12),
            blurRadius: 18,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: CustomPaint(painter: _NcgMarkPainter()),
    );
  }
}

class _NcgMarkPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final c = Offset(size.width / 2, size.height / 2);
    final r = size.shortestSide * .34;
    final circle = Paint()
      ..shader = const LinearGradient(
        colors: [Color(0xFFDCF7D8), Color(0xFF1F8A4C), Color(0xFF0F5F37)],
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
      ).createShader(Rect.fromCircle(center: c, radius: r));
    canvas.drawCircle(c, r, circle);

    final leaf = Path()
      ..moveTo(size.width * .32, size.height * .68)
      ..quadraticBezierTo(size.width * .48, size.height * .22, size.width * .72,
          size.height * .28)
      ..quadraticBezierTo(size.width * .7, size.height * .62, size.width * .32,
          size.height * .68);
    canvas.drawPath(leaf, Paint()..color = Colors.white.withOpacity(.92));

    final vein = Paint()
      ..color = ncgCanopy
      ..strokeWidth = size.width * .055
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;
    canvas.drawLine(
      Offset(size.width * .38, size.height * .66),
      Offset(size.width * .68, size.height * .32),
      vein,
    );

    final water = Paint()
      ..color = const Color(0xFF176BC2).withOpacity(.78)
      ..strokeWidth = size.width * .055
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;
    final wave = Path()
      ..moveTo(size.width * .24, size.height * .72)
      ..quadraticBezierTo(size.width * .42, size.height * .64, size.width * .58,
          size.height * .72)
      ..quadraticBezierTo(size.width * .72, size.height * .79, size.width * .86,
          size.height * .7);
    canvas.drawPath(wave, water);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class NatureBackdrop extends StatelessWidget {
  final Widget child;
  const NatureBackdrop({super.key, required this.child});

  @override
  Widget build(BuildContext context) {
    final dark = Theme.of(context).brightness == Brightness.dark;
    return Stack(
      children: [
        Positioned.fill(
          child: DecoratedBox(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: dark
                    ? const [
                        Color(0xFF07110B),
                        Color(0xFF102419),
                        Color(0xFF0B1714)
                      ]
                    : const [
                        Color(0xFFF7FBF4),
                        Color(0xFFEAF6EC),
                        Color(0xFFF8FBF5)
                      ],
              ),
            ),
          ),
        ),
        Positioned(
          top: -70,
          right: -80,
          child: _LeafWash(
              color: ncgLeaf.withOpacity(dark ? .16 : .18), size: 240),
        ),
        Positioned(
          bottom: -90,
          left: -70,
          child: _LeafWash(
              color: Colors.blue.withOpacity(dark ? .10 : .08), size: 260),
        ),
        child,
      ],
    );
  }
}

class _LeafWash extends StatelessWidget {
  final Color color;
  final double size;
  const _LeafWash({required this.color, required this.size});

  @override
  Widget build(BuildContext context) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: -0.08, end: 0.08),
      duration: const Duration(seconds: 6),
      curve: Curves.easeInOut,
      builder: (context, value, child) =>
          Transform.rotate(angle: value, child: child),
      child: Container(
        width: size,
        height: size * .58,
        decoration: BoxDecoration(
          color: color,
          borderRadius: const BorderRadius.only(
            topLeft: Radius.circular(180),
            bottomRight: Radius.circular(180),
          ),
        ),
      ),
    );
  }
}
