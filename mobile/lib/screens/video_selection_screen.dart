import 'dart:async';
import 'dart:io';

import '../widgets/video_upload_card.dart';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

class VideoSelectionScreen extends StatefulWidget {
  const VideoSelectionScreen({super.key});

  @override
  State<VideoSelectionScreen> createState() => _VideoSelectionScreenState();
}

class _VideoSelectionScreenState extends State<VideoSelectionScreen>
    with WidgetsBindingObserver {
  // Initial demo limits.
  static const int maxBytes = 30 * 1000 * 1000;
  static const Duration maxDuration = Duration(seconds: 30);

  VideoPlayerController? _controller;
  PlatformFile? _selectedFile;

  int _fileBytes = 0;
  bool _busy = false;
  bool _analysing = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);

    // Screen khulne ke baad file picker open karo.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        unawaited(_pickVideo());
      }
    });
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // App background mein jaye to video pause karo.
    if (state != AppLifecycleState.resumed) {
      unawaited(_pausePreview());
    }
  }

  Future<void> _pausePreview() async {
    try {
      await _controller?.pause();
    } catch (error) {
      debugPrint('Pause error: $error');
    }
  }

  Future<void> _disposePlayer(VideoPlayerController? player) async {
    if (player == null) return;

    try {
      await player.dispose();
    } catch (error) {
      debugPrint('Player cleanup error: $error');
    }
  }

  void _showError(String message) {
    if (!mounted) return;

    setState(() {
      _error = message;
    });
  }

  Future<void> _pickVideo() async {
    // Multiple simultaneous picker requests ko roko.
    if (_busy || _analysing || !mounted) return;

    setState(() {
      _busy = true;
      _error = null;
    });

    VideoPlayerController? pendingPlayer;

    try {
      await _pausePreview();
      if (!mounted) return;

      final file = await FilePicker.pickFile(
        type: FileType.custom,
        allowedExtensions: ['mp4'],
        compressionQuality: 0,
      );

      // Cancel karna normal action hai.
      if (!mounted || file == null) return;

      if (file.extension?.toLowerCase() != 'mp4') {
        throw const FormatException('Please select an MP4 video.');
      }

      final bytes = await file.length();
      if (!mounted) return;

      if (bytes == null || bytes <= 0) {
        throw const FormatException(
          'The file could not be read. Choose a video stored on this phone.',
        );
      }

      if (bytes > maxBytes) {
        throw const FormatException(
          'This video exceeds 30 MB. Please choose a smaller video.',
        );
      }

      // Android picker content URI ya local file URI de sakta hai.
      final uri = file.uri;

      if (uri.scheme != 'content' && uri.scheme != 'file') {
        throw const FormatException(
          'Please choose a video stored on this phone.',
        );
      }

      final player = uri.scheme == 'content'
          ? VideoPlayerController.contentUri(uri)
          : VideoPlayerController.file(File.fromUri(uri));

      pendingPlayer = player;

      await player.initialize().timeout(const Duration(seconds: 20));

      if (!mounted) return;

      if (player.value.hasError || player.value.size.isEmpty) {
        throw const FormatException(
          'This file does not contain a playable video.',
        );
      }

      final duration = player.value.duration;

      if (duration <= Duration.zero || duration > maxDuration) {
        throw const FormatException(
          'Please choose a playable video no longer than 30 seconds.',
        );
      }

      final previousPlayer = _controller;

      setState(() {
        _controller = player;
        _selectedFile = file;
        _fileBytes = bytes;
      });

      // Controller ki ownership ab screen ke paas hai.
      pendingPlayer = null;
      unawaited(_disposePlayer(previousPlayer));
    } on FormatException catch (error) {
      _showError(error.message);
    } on TimeoutException {
      _showError('The video took too long to open. Please try another MP4.');
    } catch (error) {
      debugPrint('Video selection error: $error');
      _showError('Could not open this video. Please choose another local MP4.');
    } finally {
      // Invalid selection ka player release karo.
      unawaited(_disposePlayer(pendingPlayer));

      if (mounted) {
        setState(() {
          _busy = false;
        });
      }
    }
  }

  Future<void> _togglePlayback(VideoPlayerController player) async {
    if (_busy || _analysing || player != _controller) return;

    try {
      if (player.value.isPlaying) {
        await player.pause();
      } else {
        // Video end ho chuki ho to beginning se replay karo.
        if (player.value.position >= player.value.duration) {
          await player.seekTo(Duration.zero);
        }

        if (mounted && player == _controller && !_busy && !_analysing) {
          await player.play();
        }
      }
    } catch (error) {
      debugPrint('Playback error: $error');
      _showError('Playback failed. Please select the video again.');
    }
  }

  String _formatTime(Duration duration) {
    final minutes = duration.inMinutes;
    final seconds = (duration.inSeconds % 60).toString().padLeft(2, '0');

    return '$minutes:$seconds';
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    unawaited(_disposePlayer(_controller));
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final player = _controller;

    return Scaffold(
      appBar: AppBar(title: const Text('Video preview')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(20),
          children: [
            const Text(
              'Choose your video',
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            const Text('MP4 • Up to 30 seconds • Up to 30 MB'),
            const SizedBox(height: 20),

            FilledButton.icon(
              onPressed: (_busy || _analysing) ? null : _pickVideo,
              icon: const Icon(Icons.video_library_outlined),
              label: Text(player == null ? 'Select video' : 'Change video'),
              style: FilledButton.styleFrom(
                minimumSize: const Size.fromHeight(52),
              ),
            ),

            if (_busy) ...[
              const SizedBox(height: 16),
              const LinearProgressIndicator(),
              const SizedBox(height: 8),
              const Text('Opening video...'),
            ],

            if (_error != null) ...[
              const SizedBox(height: 16),
              Text(
                _error!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            ],

            const SizedBox(height: 24),

            if (player == null && !_busy && !_analysing) ...[
              const Icon(
                Icons.video_file_outlined,
                size: 80,
                color: Color(0xFF64748B),
              ),
              const SizedBox(height: 16),
              const Text('No video selected', textAlign: TextAlign.center),
            ],

            if (player != null) ...[
              Text(
                _selectedFile?.name ?? '',
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 6),
              Text('Size: ${(_fileBytes / 1000000).toStringAsFixed(2)} MB'),
              const SizedBox(height: 16),

              ValueListenableBuilder<VideoPlayerValue>(
                valueListenable: player,
                builder: (context, value, child) {
                  if (value.hasError) {
                    return const Text(
                      'This video cannot be played. Choose another MP4.',
                    );
                  }

                  if (!value.isInitialized) {
                    return const Center(child: CircularProgressIndicator());
                  }

                  return Column(
                    children: [
                      ClipRRect(
                        borderRadius: BorderRadius.circular(16),
                        child: Container(
                          height: 280,
                          color: Colors.black,
                          child: Center(
                            child: AspectRatio(
                              aspectRatio: value.aspectRatio,
                              child: VideoPlayer(player),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(height: 12),
                      VideoProgressIndicator(
                        player,
                        allowScrubbing: !_busy && !_analysing,
                        padding: const EdgeInsets.symmetric(vertical: 12),
                      ),
                      Text(
                        '${_formatTime(value.position)} / '
                        '${_formatTime(value.duration)}',
                      ),
                      const SizedBox(height: 12),
                      FilledButton.icon(
                        onPressed: (_busy || _analysing)
                            ? null
                            : () => _togglePlayback(player),
                        icon: Icon(
                          value.isPlaying ? Icons.pause : Icons.play_arrow,
                        ),
                        label: Text(value.isPlaying ? 'Pause' : 'Play'),
                      ),
                    ],
                  );
                },
              ),
            ],
            if (!_busy && _selectedFile != null) ...[
              const SizedBox(height: 16),
              VideoUploadCard(
                key: ObjectKey(_selectedFile!),
                file: _selectedFile!,
                beforeAnalyse: _pausePreview,
                onBusyChanged: (busy) {
                  if (mounted) setState(() => _analysing = busy);
                },
              ),
            ],
          ],
        ),
      ),
    );
  }
}
