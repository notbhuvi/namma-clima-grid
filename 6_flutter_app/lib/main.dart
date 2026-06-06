import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'screens/dashboard_screen.dart';
import 'screens/map_screen.dart';
import 'screens/report_screen.dart';
import 'screens/alerts_screen.dart';
import 'services/websocket_service.dart';
import 'providers/ward_providers.dart';
import 'widgets/app_theme.dart';

void main() {
  runApp(const ProviderScope(child: NammaClimaGridApp()));
}

// ---------------------------------------------------------------------------
// App root
// ---------------------------------------------------------------------------

class NammaClimaGridApp extends ConsumerStatefulWidget {
  const NammaClimaGridApp({super.key});

  @override
  ConsumerState<NammaClimaGridApp> createState() => _NammaClimaGridAppState();
}

class _NammaClimaGridAppState extends ConsumerState<NammaClimaGridApp> {
  @override
  void initState() {
    super.initState();
    Future.microtask(() => ref.read(wsServiceProvider).connect());
  }

  @override
  void dispose() {
    ref.read(wsServiceProvider).disconnect();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final themeMode = ref.watch(themeModeProvider);
    return MaterialApp(
      title: 'NammaClimaGrid',
      debugShowCheckedModeBanner: false,
      theme: buildNcgTheme(Brightness.light),
      darkTheme: buildNcgTheme(Brightness.dark),
      themeMode: themeMode,
      home: const AppShell(),
    );
  }
}

// ---------------------------------------------------------------------------
// Shell with citizen-facing bottom navigation
// ---------------------------------------------------------------------------

class AppShell extends ConsumerStatefulWidget {
  const AppShell({super.key});

  @override
  ConsumerState<AppShell> createState() => _AppShellState();
}

class _AppShellState extends ConsumerState<AppShell> {
  int _selectedIndex = 0;

  static const _screens = [
    DashboardScreen(),
    MapScreen(),
    ReportScreen(),
    AlertsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    // Listen to alert stream to update badge count
    ref.listen<AsyncValue<List>>(alertsProvider, (_, next) {
      next.whenData((alerts) {
        if (_selectedIndex != 3) {
          // Only increment badge if not on alerts tab
          final count = alerts.length;
          ref.read(unreadAlertCountProvider.notifier).state = count;
        }
      });
    });

    final unreadCount = ref.watch(unreadAlertCountProvider);

    return Scaffold(
      body: IndexedStack(
        index: _selectedIndex,
        children: _screens,
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (i) {
          setState(() => _selectedIndex = i);
          // Clear badge when navigating to alerts
          if (i == 3) {
            ref.read(unreadAlertCountProvider.notifier).state = 0;
          }
        },
        destinations: [
          const NavigationDestination(
            icon: Icon(Icons.dashboard_outlined),
            selectedIcon: Icon(Icons.dashboard),
            label: 'Dashboard',
          ),
          const NavigationDestination(
            icon: Icon(Icons.map_outlined),
            selectedIcon: Icon(Icons.map),
            label: 'Map',
          ),
          const NavigationDestination(
            icon: Icon(Icons.edit_note_outlined),
            selectedIcon: Icon(Icons.edit_note),
            label: 'Report',
          ),
          NavigationDestination(
            icon: unreadCount > 0
                ? Badge(
                    label: Text('$unreadCount'),
                    child: const Icon(Icons.notifications_outlined))
                : const Icon(Icons.notifications_outlined),
            selectedIcon: unreadCount > 0
                ? Badge(
                    label: Text('$unreadCount'),
                    child: const Icon(Icons.notifications))
                : const Icon(Icons.notifications),
            label: 'Alerts',
          ),
        ],
      ),
    );
  }
}
