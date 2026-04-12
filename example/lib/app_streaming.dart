// TODO maybe integrate into non-streaming all

import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_ondevice_asr/common/result.dart';
import 'package:flutter_ondevice_asr/flutter_ondevice_asr.dart';
import 'package:flutter_ondevice_asr/model/transcription_result.dart';
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
      title: 'Whisper Streaming Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const StreamingTranscriptionPage(),
    );
  }
}

class StreamingTranscriptionPage extends StatefulWidget {
  const StreamingTranscriptionPage({super.key});

  @override
  State<StreamingTranscriptionPage> createState() => _StreamingTranscriptionPageState();
}

class _StreamingTranscriptionPageState extends State<StreamingTranscriptionPage> {
  final AudioRecorder _audioRecorder = AudioRecorder();
  late final Transcriber _whisper;

  StreamingTranscriber? _streaming;
  StreamSubscription<TranscriptionResult>? _transcriptionSubscription;

  bool _isRecording = false;
  bool _isLoading = false;
  bool _modelsLoaded = false;

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

  // Immutable VAD parameters (set at model load time)
  double _vadThreshold = 0.5;
  double _eosMinSilence = 300;  // milliseconds

  // Mutable session parameters (changeable between recordings)
  bool _enablePartials = true;
  double _minPartialDuration = 1000;  // milliseconds
  int _maxSegmentDuration = 10000;  // milliseconds
  final TextEditingController _maxSegmentController = TextEditingController(text: '10000');

  double _loadingProgress = 0.0;
  String _loadingStep = '';
  String _statusMessage = 'Press Load Models to begin';
  String _logMessages = '';
  String _transcription = '';  // Unified transcription (finals + current partial)
  String _currentPartial = '';  // Current partial being updated
  DateTime? _recordingStartTime;
  int _chunksProcessed = 0;
  double _audioEnergyLevel = 0.0;
  int _totalAudioSamples = 0;  // Track total audio samples for RTF calculation
  int _totalProcessingTimeMs = 0;  // Accumulate processing time from transcriptions

  // Audio streaming state
  StreamSubscription<Uint8List>? _audioStreamSubscription;

  @override
  void initState() {
    super.initState();
  }

  void _addLog(String message) {
    setState(() {
      final timestamp = DateTime.now().toString().substring(11, 23);
      _logMessages += '[$timestamp] $message\n';
    });
  }

  String _buildTranscriptionDisplay() {
    if (_transcription.isEmpty && _currentPartial.isEmpty) {
      return '(transcriptions will appear here)';
    }

    // Combine finals and current partial
    String display = _transcription;
    if (_currentPartial.isNotEmpty) {
      if (display.isNotEmpty) {
        display += '\n';
      }
      display += '[PARTIAL] $_currentPartial';
    }
    return display;
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
      _whisper = Transcriber.getInstance(TranscriberType.whisper);

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
      await _whisper.loadModel(
        modelDirectory: 'assets/transcribers/whisper/models/whisper_tiny/default_int8',
        languageCode: _selectedLanguage,
      );

      final duration = DateTime.now().difference(startTime);
      _addLog('Models loaded in ${duration.inMilliseconds}ms');

      // Initialize streaming transcriber with all parameters
      _streaming = await StreamingTranscriber.create(
        transcriber: _whisper,
        vadThreshold: _vadThreshold,
        eosMinSilence: _eosMinSilence.toInt(),
        sampleRate: 16000,
        enablePartials: _enablePartials,
        minPartialDuration: _minPartialDuration.toInt(),
        maxSegmentDuration: _maxSegmentDuration,
      );

      // Listen to transcription stream
      _transcriptionSubscription = _streaming!.transcriptionStream.listen((resultResult) {
        final result = (resultResult as Ok<TranscriptionResult>).value;
        setState(() {
          if (result.isFinal) {
            // Add final segment on new line
            if (_transcription.isNotEmpty) {
              _transcription += '\n';
            }
            _transcription += result.text;
            _currentPartial = '';  // Clear partial

            _addLog('FINAL (${result.durationInSeconds.toStringAsFixed(1)}s): ${result.text}');
          } else {
            // Update current partial
            _currentPartial = result.text;
            _addLog('PARTIAL (${result.durationInSeconds.toStringAsFixed(1)}s): ${result.text}');
          }
        });
      });

      _addLog('Streaming transcriber initialized');

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

  Future<void> _startStreamingRecording() async {
    try {
      await _requestMicrophonePermission();

      if (await _audioRecorder.hasPermission()) {
        // Configure streaming with current settings
        _streaming!.configure(
          enablePartials: _enablePartials,
          minPartialDuration: _minPartialDuration.toInt(),
          maxSegmentDuration: _maxSegmentDuration,
        );

        // Reset state to start fresh
        _streaming!.reset();

        // Listen to transcription stream (reuse existing subscription if present)
        _transcriptionSubscription ??= _streaming!.transcriptionStream.listen((result) {
          setState(() {
            if (result.isFinal) {
              // Add final segment on new line
              if (_transcription.isNotEmpty) {
                _transcription += '\n';
              }
              _transcription += result.text;
              _currentPartial = '';  // Clear partial

              _addLog('FINAL (${result.durationInSeconds.toStringAsFixed(1)}s): ${result.text}');
            } else {
              // Update current partial
              _currentPartial = result.text;
              _addLog('PARTIAL (${result.durationInSeconds.toStringAsFixed(1)}s): ${result.text}');
            }
          });
          });

        _addLog('Streaming configured: partials=$_enablePartials, minPartial=${_minPartialDuration.toInt()}ms, maxSegment=${_maxSegmentDuration}ms');

        // Configure for streaming: 16kHz mono PCM
        final stream = await _audioRecorder.startStream(
          const RecordConfig(
            encoder: AudioEncoder.pcm16bits,
            numChannels: 1,
            sampleRate: 16000,
          ),
        );

        _addLog('Started streaming recording');
        _recordingStartTime = DateTime.now();

        // Subscribe to audio stream and feed to transcriber
        _audioStreamSubscription = stream.listen((audioChunk) async {
          // Convert Int16 PCM to Float32 normalized [-1, 1]
          final int16Data = Uint8List.fromList(audioChunk).buffer.asInt16List();
          final float32Data = Float32List(int16Data.length);
          for (int i = 0; i < int16Data.length; i++) {
            float32Data[i] = int16Data[i] / 32768.0;
          }

          // Calculate energy level for feedback
          double energy = 0.0;
          for (var sample in float32Data) {
            energy += sample * sample;
          }
          energy = (energy / float32Data.length).clamp(0.0, 1.0);

          setState(() {
            _chunksProcessed++;
            _audioEnergyLevel = energy;
            _totalAudioSamples += float32Data.length;
          });

          // Feed to streaming transcriber and measure processing time
          final processingStart = DateTime.now();
          await _streaming?.processAudioChunk(float32Data);
          final processingTime = DateTime.now().difference(processingStart).inMilliseconds;
          _totalProcessingTimeMs += processingTime;
        });

        setState(() {
          _isRecording = true;
          _statusMessage = 'Recording... Speak into microphone';
          _transcription = '';
          _currentPartial = '';
          _chunksProcessed = 0;
          _audioEnergyLevel = 0.0;
          _totalAudioSamples = 0;
          _totalProcessingTimeMs = 0;
        });
      }
    } catch (e) {
      _addLog('Error starting recording: $e');
      setState(() {
        _statusMessage = 'Error starting recording: $e';
      });
    }
  }

  Future<void> _stopStreamingRecording() async {
    try {
      // Cancel audio stream subscription
      await _audioStreamSubscription?.cancel();
      _audioStreamSubscription = null;

      await _audioRecorder.stop();

      final recordingDuration = _recordingStartTime != null
          ? DateTime.now().difference(_recordingStartTime!).inSeconds
          : 0;

      // Calculate RTF (Real-Time Factor)
      const sampleRate = 16000;
      final audioLengthSeconds = _totalAudioSamples / sampleRate;
      final processingTimeSeconds = _totalProcessingTimeMs / 1000.0;
      final rtf = audioLengthSeconds > 0 ? processingTimeSeconds / audioLengthSeconds : 0.0;

      _addLog('Recording stopped. Duration: ${recordingDuration}s');
      _addLog('Audio length: ${audioLengthSeconds.toStringAsFixed(2)}s');
      _addLog('Processing time: ${processingTimeSeconds.toStringAsFixed(2)}s');
      _addLog('RTF (Real-Time Factor): ${rtf.toStringAsFixed(3)}x');
      if (rtf < 1.0 && rtf > 0) {
        _addLog('  → ${(1.0 / rtf).toStringAsFixed(2)}x faster than real-time');
      } else if (rtf >= 1.0) {
        _addLog('  → Slower than real-time');
      }

      // Flush any remaining audio in streaming buffer
      await _streaming?.flush();

      setState(() {
        _isRecording = false;
        _statusMessage = 'Recording complete. RTF: ${rtf.toStringAsFixed(3)}x';
        _recordingStartTime = null;
      });
    } catch (e) {
      _addLog('Error stopping recording: $e');
      setState(() {
        _statusMessage = 'Error stopping recording: $e';
      });
    }
  }

  Future<void> _clearTranscriptions() async {
    setState(() {
      _transcription = '';
      _currentPartial = '';
      _logMessages = '';
    });
    _streaming?.reset();
  }

  @override
  void dispose() {
    _audioRecorder.dispose();
    _transcriptionSubscription?.cancel();
    _audioStreamSubscription?.cancel();
    _streaming?.dispose();
    _maxSegmentController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Whisper Streaming Demo'),
      ),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Compact header with buttons
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Column(
              children: [
                // Status row - compact
                if (_isRecording)
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: Colors.red[50],
                      borderRadius: BorderRadius.circular(4),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(Icons.mic, size: 14, color: Colors.red),
                        const SizedBox(width: 6),
                        Text(
                          'Recording',
                          style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                color: Colors.red,
                                fontWeight: FontWeight.bold,
                              ),
                        ),
                        const SizedBox(width: 8),
                        // Audio level indicator
                        Container(
                          width: 80,
                          height: 4,
                          decoration: BoxDecoration(
                            color: Colors.grey[300],
                            borderRadius: BorderRadius.circular(2),
                          ),
                          child: FractionallySizedBox(
                            widthFactor: (_audioEnergyLevel * 100).clamp(0.0, 1.0),
                            alignment: Alignment.centerLeft,
                            child: Container(
                              decoration: BoxDecoration(
                                color: _audioEnergyLevel > 0.0005 ? Colors.green : Colors.grey,
                                borderRadius: BorderRadius.circular(2),
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(width: 6),
                        Text(
                          '${(_audioEnergyLevel * 1000).toStringAsFixed(1)}',
                          style: Theme.of(context).textTheme.bodySmall?.copyWith(fontSize: 10),
                        ),
                      ],
                    ),
                  ),
                if (_isLoading)
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(value: _loadingProgress, strokeWidth: 2),
                        ),
                        const SizedBox(width: 8),
                        Text(_loadingStep, style: Theme.of(context).textTheme.bodySmall),
                      ],
                    ),
                  ),
                const SizedBox(height: 6),

                // Language selection (only show BEFORE models loaded)
                if (!_modelsLoaded && !_isLoading) ...[
                  const Text('Language:', style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 4),
                  DropdownButtonFormField<String>(
                    initialValue: _selectedLanguage,
                    decoration: const InputDecoration(
                      isDense: true,
                      contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      border: OutlineInputBorder(),
                    ),
                    style: const TextStyle(fontSize: 12, color: Colors.black),
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
                  const SizedBox(height: 8),
                ],

                // Immutable VAD parameters (only show BEFORE models loaded)
                if (!_modelsLoaded && !_isLoading) ...[
                  const Text('VAD Settings (immutable after loading):', style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 4),

                  // VAD Threshold Slider
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'VAD Threshold: ${_vadThreshold.toStringAsFixed(2)}',
                        style: const TextStyle(fontSize: 11),
                      ),
                      Slider(
                        value: _vadThreshold,
                        min: 0.1,
                        max: 0.9,
                        divisions: 8,
                        label: _vadThreshold.toStringAsFixed(2),
                        onChanged: (value) {
                          setState(() {
                            _vadThreshold = value;
                          });
                        },
                      ),
                    ],
                  ),

                  // EOS Min Silence Slider
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'End of Speech Silence: ${_eosMinSilence.toInt()}ms',
                        style: const TextStyle(fontSize: 11),
                      ),
                      Slider(
                        value: _eosMinSilence,
                        min: 200,
                        max: 1500,
                        divisions: 13,
                        label: '${_eosMinSilence.toInt()}ms',
                        onChanged: (value) {
                          setState(() {
                            _eosMinSilence = value;
                          });
                        },
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                ],

                // Mutable session parameters (only show when models loaded and not recording)
                if (_modelsLoaded && !_isRecording) ...[
                  const Text('Session Settings:', style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 4),

                  // Enable Partials Toggle
                  Row(
                    children: [
                      Checkbox(
                        value: _enablePartials,
                        onChanged: (value) {
                          setState(() {
                            _enablePartials = value ?? true;
                          });
                        },
                      ),
                      const Text('Enable Partials', style: TextStyle(fontSize: 12)),
                      const Spacer(),
                      if (!_enablePartials)
                        Text(
                          '(Finals only)',
                          style: TextStyle(fontSize: 10, color: Colors.grey[600]),
                        ),
                    ],
                  ),

                  // Min Partial Duration Slider
                  if (_enablePartials)
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Min Partial Duration: ${_minPartialDuration.toInt()}ms',
                          style: const TextStyle(fontSize: 11),
                        ),
                        Slider(
                          value: _minPartialDuration,
                          min: 300,
                          max: 3000,
                          divisions: 27,
                          label: '${_minPartialDuration.toInt()}ms',
                          onChanged: (value) {
                            setState(() {
                              _minPartialDuration = value;
                            });
                          },
                        ),
                      ],
                    ),

                  // Max Segment Duration Text Box
                  Row(
                    children: [
                      const Text('Max Segment Duration (ms):', style: TextStyle(fontSize: 11)),
                      const SizedBox(width: 8),
                      SizedBox(
                        width: 80,
                        child: TextField(
                          controller: _maxSegmentController,
                          keyboardType: TextInputType.number,
                          style: const TextStyle(fontSize: 12),
                          decoration: const InputDecoration(
                            isDense: true,
                            contentPadding: EdgeInsets.symmetric(horizontal: 8, vertical: 8),
                            border: OutlineInputBorder(),
                          ),
                          onChanged: (value) {
                            final parsed = int.tryParse(value);
                            if (parsed != null && parsed > 0) {
                              setState(() {
                                _maxSegmentDuration = parsed;
                              });
                            }
                          },
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 6),
                ],

                // Buttons in a row
                Row(
                  children: [
                    // Load Models button (only show if models not loaded)
                    if (!_modelsLoaded)
                      Expanded(
                        child: ElevatedButton(
                          onPressed: _isLoading ? null : _initializeWhisper,
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(vertical: 8),
                            backgroundColor: Colors.blue,
                            foregroundColor: Colors.white,
                          ),
                          child: const Text('Load', style: TextStyle(fontSize: 12)),
                        ),
                      ),
                    if (!_modelsLoaded) const SizedBox(width: 6),

                    // Record button
                    Expanded(
                      child: ElevatedButton(
                        onPressed: !_modelsLoaded || _isLoading
                            ? null
                            : () {
                                if (_isRecording) {
                                  _stopStreamingRecording();
                                } else {
                                  _startStreamingRecording();
                                }
                              },
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 8),
                          backgroundColor: _isRecording ? Colors.red : null,
                          foregroundColor: _isRecording ? Colors.white : null,
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(_isRecording ? Icons.stop : Icons.mic, size: 14),
                            const SizedBox(width: 4),
                            Text(
                              _isRecording ? 'Stop' : 'Start',
                              style: const TextStyle(fontSize: 12),
                            ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(width: 6),

                    // Clear button
                    Expanded(
                      child: ElevatedButton(
                        onPressed: !_modelsLoaded || _isLoading || _isRecording
                            ? null
                            : _clearTranscriptions,
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 8),
                        ),
                        child: const Text('Clear', style: TextStyle(fontSize: 12)),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),

          // Unified transcription window (finals + current partial)
          if (_modelsLoaded)
            Expanded(
              flex: 2,
              child: Card(
                margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                child: Padding(
                  padding: const EdgeInsets.all(10.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Transcription',
                        style: Theme.of(context).textTheme.titleSmall?.copyWith(
                              fontWeight: FontWeight.bold,
                              fontSize: 12,
                            ),
                      ),
                      const SizedBox(height: 4),
                      Expanded(
                        child: Container(
                          width: double.infinity,
                          padding: const EdgeInsets.all(8.0),
                          decoration: BoxDecoration(
                            color: Colors.grey[50],
                            borderRadius: BorderRadius.circular(4),
                            border: Border.all(color: Colors.grey[300]!),
                          ),
                          child: SingleChildScrollView(
                            child: SelectableText(
                              _buildTranscriptionDisplay(),
                              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                color: Colors.black87,
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

          // Logs - expanded (take 3/5 of space)
          Expanded(
            flex: 3,
            child: Container(
            decoration: BoxDecoration(
              color: Colors.grey[200],
              border: Border(top: BorderSide(color: Colors.grey[300]!)),
            ),
            child: Padding(
              padding: const EdgeInsets.all(8.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        'Logs',
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                              fontWeight: FontWeight.bold,
                              fontSize: 11,
                            ),
                      ),
                      InkWell(
                        onTap: () {
                          setState(() {
                            _logMessages = '';
                          });
                        },
                        child: const Icon(Icons.clear, size: 14),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Expanded(
                    child: Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(4.0),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(3),
                      ),
                      child: SingleChildScrollView(
                        child: SelectableText(
                          _logMessages.isEmpty ? 'No logs' : _logMessages,
                          style: const TextStyle(
                            fontFamily: 'monospace',
                            fontSize: 9,
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
