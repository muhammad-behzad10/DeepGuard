import 'package:flutter/material.dart';

import 'screens/video_selection_screen.dart';
import 'screens/scan_history_screen.dart';
import 'screens/connection_settings_screen.dart';
import 'services/api_service.dart';
import 'services/connection_settings_store.dart';
import 'services/scan_history_store.dart';
import 'widgets/backend_status_card.dart';

void main() {
  runApp(const MyApp());
}

// App ka naam, theme aur starting screen.
class MyApp extends StatelessWidget {
  final ConnectionSettingsStore? settingsStore;
  final ScanHistoryStore? historyStore;
  final HealthChecker? healthChecker;
  const MyApp({
    super.key,
    this.settingsStore,
    this.historyStore,
    this.healthChecker,
  });

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'DeepGuard',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0F766E)),
        scaffoldBackgroundColor: const Color(0xFFF5F7FA),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFFF5F7FA),
          foregroundColor: Color(0xFF14243B),
          elevation: 0,
          centerTitle: false,
        ),
      ),
      home: HomeScreen(
        settingsStore: settingsStore,
        historyStore: historyStore,
        healthChecker: healthChecker,
      ),
    );
  }
}

// Bottom navigation aur selected screen ko manage karta hai.
class HomeScreen extends StatefulWidget {
  final ConnectionSettingsStore? settingsStore;
  final ScanHistoryStore? historyStore;
  final HealthChecker? healthChecker;
  const HomeScreen({
    super.key,
    this.settingsStore,
    this.historyStore,
    this.healthChecker,
  });

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int selectedIndex = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'DeepGuard',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        actions: [
          IconButton(
            tooltip: 'Connection settings',
            icon: const Icon(Icons.settings_outlined),
            onPressed: () {
              Navigator.of(context).push<void>(
                MaterialPageRoute(
                  builder: (_) => ConnectionSettingsScreen(
                    store: widget.settingsStore,
                    healthChecker: widget.healthChecker,
                  ),
                ),
              );
            },
          ),
          const Padding(
            padding: EdgeInsets.only(right: 16),
            child: Chip(label: Text('Prototype')),
          ),
        ],
      ),
      body: SafeArea(
        child: IndexedStack(
          index: selectedIndex,
          children: [
            ScanScreen(
              settingsStore: widget.settingsStore,
              healthChecker: widget.healthChecker,
            ),
            ScanHistoryScreen(store: widget.historyStore),
            const MessageScreen(
              icon: Icons.info_outline,
              title: 'About DeepGuard',
              message:
                  'DeepGuard is a final-year research project focused '
                  'on identifying facial manipulation in videos. It uses a pretrained '
                  'DeepfakeBench Xception model and OpenCV YuNet face detection. '
                  'Predictions can be wrong. It assesses sampled face frames; '
                  'audio and identity tracking are not analysed.',
            ),
          ],
        ),
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: selectedIndex,
        onDestinationSelected: (int index) {
          setState(() {
            selectedIndex = index;
          });
        },
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.video_library_outlined),
            selectedIcon: Icon(Icons.video_library),
            label: 'Scan',
          ),
          NavigationDestination(icon: Icon(Icons.history), label: 'History'),
          NavigationDestination(
            icon: Icon(Icons.info_outline),
            selectedIcon: Icon(Icons.info),
            label: 'About',
          ),
        ],
      ),
    );
  }
}

// Video scan ki introductory screen.
class ScanScreen extends StatelessWidget {
  final ConnectionSettingsStore? settingsStore;
  final HealthChecker? healthChecker;
  const ScanScreen({super.key, this.settingsStore, this.healthChecker});

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(20),
      children: [
        BackendStatusCard(store: settingsStore, healthChecker: healthChecker),
        const SizedBox(height: 16),
        Container(
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            color: const Color(0xFF14243B),
            borderRadius: BorderRadius.circular(24),
          ),
          child: const Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(Icons.shield_outlined, size: 48, color: Color(0xFF5EEAD4)),
              SizedBox(height: 20),
              Text(
                'Deepfake video detection',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 28,
                  fontWeight: FontWeight.bold,
                  height: 1.2,
                ),
              ),
              SizedBox(height: 12),
              Text(
                'Assess facial manipulation in videos.',
                style: TextStyle(
                  color: Color(0xFFCBD5E1),
                  fontSize: 16,
                  height: 1.5,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 24),
        const Text(
          'Start a video scan',
          style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 8),
        const Text(
          'Choose a video with a clearly visible face.',
          style: TextStyle(color: Color(0xFF475569), fontSize: 15),
        ),
        const SizedBox(height: 20),

        FilledButton.icon(
          onPressed: () {
            Navigator.of(context).push(
              MaterialPageRoute<void>(
                builder: (context) => const VideoSelectionScreen(),
              ),
            );
          },
          icon: const Icon(Icons.video_library_outlined),
          label: const Text('Select video'),
          style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(52)),
        ),
        const SizedBox(height: 28),
        const Text(
          'How it works',
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        const Card(
          child: Column(
            children: [
              ListTile(
                leading: Icon(Icons.video_file_outlined),
                title: Text('1. Choose a video'),
                subtitle: Text('Select a supported video from your phone.'),
              ),
              Divider(height: 1),
              ListTile(
                leading: Icon(Icons.analytics_outlined),
                title: Text('2. Review the assessment'),
                subtitle: Text('View the analysis result and model score.'),
              ),
              Divider(height: 1),
              ListTile(
                leading: Icon(Icons.history),
                title: Text('3. Revisit saved results'),
                subtitle: Text(
                  'Open History to review or delete results saved on this device.',
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

// History aur About ke liye reusable informational layout.
class MessageScreen extends StatelessWidget {
  final IconData icon;
  final String title;
  final String message;

  const MessageScreen({
    super.key,
    required this.icon,
    required this.title,
    required this.message,
  });

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 64),
      children: [
        Icon(icon, size: 72, color: Theme.of(context).colorScheme.primary),
        const SizedBox(height: 24),
        Text(
          title,
          textAlign: TextAlign.center,
          style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        Text(
          message,
          textAlign: TextAlign.center,
          style: const TextStyle(
            color: Color(0xFF475569),
            fontSize: 16,
            height: 1.6,
          ),
        ),
      ],
    );
  }
}
