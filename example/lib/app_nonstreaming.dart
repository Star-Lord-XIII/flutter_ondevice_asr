import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_ondevice_asr/flutter_ondevice_asr.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Whisper Transcription Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const TranscriptionDemoPage(),
    );
  }
}

class TranscriptionDemoPage extends StatefulWidget {
  const TranscriptionDemoPage({super.key});

  @override
  State<TranscriptionDemoPage> createState() => _TranscriptionDemoPageState();
}

class _TranscriptionDemoPageState extends State<TranscriptionDemoPage> {
  final AudioRecorder _audioRecorder = AudioRecorder();
  late final WhisperTranscriber _whisper;

  bool _isRecording = false;
  bool _isLoading = false;
  bool _modelsLoaded = false;
  bool _isTranscribing = false;
  String _transcription = '';
  String _statusMessage = 'Press Load Models to begin';
  String? _recordingPath;
  String _logMessages = '';
  DateTime? _recordingStartTime;
  double _recordingDuration = 0.0;
  double _loadingProgress = 0.0;
  String _loadingStep = '';

  // Language selection
  String _selectedLanguage = 'en';
  final List<Map<String, String>> _supportedLanguages = [
    {'code': 'en', 'name': 'English'},
    {'code': 'es', 'name': 'Spanish'},
    {'code': 'fr', 'name': 'French'},
    {'code': 'de', 'name': 'German'},
    {'code': 'it', 'name': 'Italian'},
    {'code': 'pt', 'name': 'Portuguese'},
    {'code': 'nl', 'name': 'Dutch'},
    {'code': 'ru', 'name': 'Russian'},
    {'code': 'zh', 'name': 'Chinese'},
    {'code': 'ja', 'name': 'Japanese'},
    {'code': 'ko', 'name': 'Korean'},
    {'code': 'ar', 'name': 'Arabic'},
    {'code': 'hi', 'name': 'Hindi'},
    {'code': 'pl', 'name': 'Polish'},
    {'code': 'tr', 'name': 'Turkish'},
  ];

  @override
  void initState() {
    super.initState();
    // Don't load models automatically - user will trigger it
  }

  void _addLog(String message) {
    setState(() {
      final timestamp = DateTime.now().toString().substring(11, 23);
      _logMessages += '[$timestamp] $message\n';
    });
  }

  Future<void> _initializeWhisper() async {
    try {
      setState(() {
        _isLoading = true;
        _loadingProgress = 0.0;
        _loadingStep = 'Starting...';
        _statusMessage = 'Loading Whisper models...';
      });

      _addLog('Starting model loading with language: $_selectedLanguage');
      final startTime = DateTime.now();

      // Initialize WhisperTranscriber with selected language
      _whisper = WhisperTranscriber(
        modelDirectory: 'assets/transcribers/whisper/models/default_int8',
        language: _selectedLanguage,
        verbose: false,
      );

      // Simulate progress tracking (ONNX Runtime doesn't expose real progress)
      setState(() {
        _loadingProgress = 0.2;
        _loadingStep = 'Loading encoder...';
      });

      await Future.delayed(const Duration(milliseconds: 100));

      setState(() {
        _loadingProgress = 0.4;
        _loadingStep = 'Loading decoder...';
      });

      // Actually load the models
      await _whisper.loadModels();

      final duration = DateTime.now().difference(startTime);
      _addLog('Models loaded in ${duration.inMilliseconds}ms');

      setState(() {
        _isLoading = false;
        _modelsLoaded = true;
        _loadingProgress = 1.0;
        _loadingStep = 'Complete';
        _statusMessage = 'Ready to record';
      });
    } catch (e) {
      _addLog('Error loading models: $e');
      setState(() {
        _isLoading = false;
        _modelsLoaded = false;
        _statusMessage = 'Error loading models: $e';
      });
    }
  }

  Future<void> _requestMicrophonePermission() async {
    // On macOS, permission_handler has limited support
    // macOS will prompt for permission automatically when recording starts
    if (Platform.isMacOS) {
      return;
    }

    final status = await Permission.microphone.request();
    if (!status.isGranted) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Microphone permission is required to record audio'),
          ),
        );
      }
    }
  }

  Future<void> _startRecording() async {
    try {
      await _requestMicrophonePermission();

      if (await _audioRecorder.hasPermission()) {
        final directory = await getTemporaryDirectory();
        final path = '${directory.path}/recording_${DateTime.now().millisecondsSinceEpoch}.wav';

        // Configure for 16-bit PCM mono WAV (as expected by Audio.loadAudio)
        await _audioRecorder.start(
          const RecordConfig(
            encoder: AudioEncoder.wav,
            numChannels: 1,
            sampleRate: 16000,
            bitRate: 256000,
          ),
          path: path,
        );

        _addLog('Started recording');
        setState(() {
          _isRecording = true;
          _statusMessage = 'Recording... Tap to stop';
          _recordingPath = path;
          _transcription = '';
          _recordingStartTime = DateTime.now();
        });
      }
    } catch (e) {
      _addLog('Error starting recording: $e');
      setState(() {
        _statusMessage = 'Error starting recording: $e';
      });
    }
  }

  Future<void> _stopRecording() async {
    try {
      final path = await _audioRecorder.stop();

      if (_recordingStartTime != null) {
        _recordingDuration = DateTime.now().difference(_recordingStartTime!).inMilliseconds / 1000.0;
        _addLog('Recording stopped. Duration: ${_recordingDuration.toStringAsFixed(2)}s');
      }

      setState(() {
        _isRecording = false;
        _statusMessage = 'Recording saved. Tap Transcribe to process.';
        if (path != null) {
          _recordingPath = path;
        }
        _recordingStartTime = null;
      });
    } catch (e) {
      _addLog('Error stopping recording: $e');
      setState(() {
        _statusMessage = 'Error stopping recording: $e';
      });
    }
  }

  Future<void> _transcribeRecording() async {
    if (_recordingPath == null) {
      setState(() {
        _statusMessage = 'No recording found. Please record first.';
      });
      return;
    }

    try {
      setState(() {
        _isTranscribing = true;
        _statusMessage = 'Transcribing...';
      });

      _addLog('Starting transcription...');
      final startTime = DateTime.now();

      final result = await _whisper.transcribeFile(_recordingPath!);

      final duration = DateTime.now().difference(startTime);
      _addLog('Transcription completed in ${duration.inMilliseconds}ms');

      setState(() {
        _isTranscribing = false;
        _transcription = result.text;
        _statusMessage = 'Transcription complete!';
      });

      // Clean up the recording file
      try {
        await File(_recordingPath!).delete();
        _recordingPath = null;
      } catch (e) {
        debugPrint('Error deleting recording file: $e');
      }
    } catch (e) {
      _addLog('Error transcribing: $e');
      setState(() {
        _isTranscribing = false;
        _statusMessage = 'Error transcribing: $e';
      });
    }
  }

  @override
  void dispose() {
    _audioRecorder.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Whisper Transcription Demo'),
      ),
      body: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Status card and buttons with padding
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Status card
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      children: [
                        if (_isLoading)
                          Column(
                            children: [
                              CircularProgressIndicator(value: _loadingProgress),
                              const SizedBox(height: 8),
                              Text(_loadingStep, style: Theme.of(context).textTheme.bodySmall),
                            ],
                          )
                        else if (_isTranscribing)
                          const CircularProgressIndicator()
                        else if (_isRecording)
                          const Icon(Icons.mic, size: 48, color: Colors.red)
                        else if (_recordingPath != null)
                          const Icon(Icons.audio_file, size: 48, color: Colors.green)
                        else if (_modelsLoaded)
                          const Icon(Icons.mic_none, size: 48, color: Colors.grey)
                        else
                          const Icon(Icons.download, size: 48, color: Colors.blue),
                        const SizedBox(height: 8),
                        Text(
                          _statusMessage,
                          style: Theme.of(context).textTheme.titleMedium,
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 24),

                // Language selection (only show if models not loaded)
                if (!_modelsLoaded)
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Language:',
                        style: Theme.of(context).textTheme.titleSmall?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 8),
                      DropdownButtonFormField<String>(
                        initialValue: _selectedLanguage,
                        decoration: const InputDecoration(
                          contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 12),
                          border: OutlineInputBorder(),
                        ),
                        items: _supportedLanguages.map((lang) {
                          return DropdownMenuItem<String>(
                            value: lang['code'],
                            child: Text('${lang['name']} (${lang['code']})'),
                          );
                        }).toList(),
                        onChanged: (value) {
                          setState(() {
                            _selectedLanguage = value ?? 'en';
                          });
                        },
                      ),
                      const SizedBox(height: 16),
                    ],
                  ),

                // Load Models button (only show if models not loaded)
                if (!_modelsLoaded)
                  ElevatedButton.icon(
                    onPressed: _isLoading ? null : _initializeWhisper,
                    icon: const Icon(Icons.download),
                    label: const Text('Load Models'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.all(16),
                      backgroundColor: Colors.blue,
                      foregroundColor: Colors.white,
                    ),
                  ),

                if (!_modelsLoaded)
                  const SizedBox(height: 16),

                // Record button
                ElevatedButton.icon(
                  onPressed: !_modelsLoaded || _isLoading || _isTranscribing
                      ? null
                      : () {
                          if (_isRecording) {
                            _stopRecording();
                          } else {
                            _startRecording();
                          }
                        },
                  icon: Icon(_isRecording ? Icons.stop : Icons.mic),
                  label: Text(_isRecording ? 'Stop Recording' : 'Start Recording'),
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.all(16),
                    backgroundColor: _isRecording ? Colors.red : null,
                    foregroundColor: _isRecording ? Colors.white : null,
                  ),
                ),

                const SizedBox(height: 16),

                // Transcribe button
                ElevatedButton.icon(
                  onPressed: !_modelsLoaded || _isLoading || _isTranscribing || _recordingPath == null
                      ? null
                      : _transcribeRecording,
                  icon: const Icon(Icons.text_fields),
                  label: const Text('Transcribe'),
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.all(16),
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 8),

          // Transcription result - full width (MOVED ABOVE LOGS)
          if (_transcription.isNotEmpty)
            Card(
              margin: const EdgeInsets.only(bottom: 8),
              shape: const RoundedRectangleBorder(borderRadius: BorderRadius.zero),
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Transcription:',
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    SelectableText(
                      _transcription,
                      style: Theme.of(context).textTheme.bodyLarge,
                    ),
                  ],
                ),
              ),
            ),

          // Logs - fixed height (smaller than before)
          SizedBox(
            height: 200, // Fixed height - about 2/3 of original
            child: Card(
              margin: EdgeInsets.zero,
              shape: const RoundedRectangleBorder(borderRadius: BorderRadius.zero),
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          'Logs:',
                          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        IconButton(
                          icon: const Icon(Icons.clear, size: 20),
                          onPressed: () {
                            setState(() {
                              _logMessages = '';
                            });
                          },
                          tooltip: 'Clear logs',
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Expanded(
                      child: Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(8.0),
                        decoration: BoxDecoration(
                          color: Colors.grey[100],
                          borderRadius: BorderRadius.circular(4),
                          border: Border.all(color: Colors.grey[300]!),
                        ),
                        child: SingleChildScrollView(
                          child: SelectableText(
                            _logMessages.isEmpty ? 'No logs yet' : _logMessages,
                            style: const TextStyle(
                              fontFamily: 'monospace',
                              fontSize: 12,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
