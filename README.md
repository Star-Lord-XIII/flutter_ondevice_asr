# Flutter On-Device ASR

A Flutter library for on-device automatic speech recognition (ASR):
- Model-agnostic architecture supporting arbitrary ASR models
- Streaming and non-streaming transcription
- ONNX Runtime-based inference (different quantization schemes)

## Library Structure

**Core Abstractions:**
- `Transcriber`: Base interface for all ASR models (defines `loadModels()`, `transcribe()`, `transcribeFile()`)
- `StreamingTranscriber`: Interface for streaming transcription
- `TranscriptionResult`: Unified result format with text and optional confidence scores
- `OnnxConfig`: Runtime configuration for ONNX models

**Existing Transcriber Implementations:**
- `WhisperTranscriber`: Whisper model implementation in `lib/models/whisper/`
- Additional models can be added by implementing the `Transcriber` interface

**Assets:**
- `assets/transcribers/whisper/models/`: Whisper ONNX model files (super_encoder, decoders, configs)
- `assets/transcribers/whisper/tokenizer/`: Whisper tokenizer vocabulary (multilingual and English-only)
- `assets/vad/silero_vad/`: Voice activity detection model
- `assets/audio/`: Test audio samples

## Details on Whisper Implementation

The Whisper implementation uses a modified architecture for efficiency on phones:

### Tokenization

* Uses `WhisperTokenizer` (`lib/models/whisper/whisper_tokenizer.dart`) for text encoding/decoding.
* Works for en-only and multilingual whisper models

### Architecture 

**Super-Encoder:** The preprocessing stage (log-mel spectrogram conversion) is merged into the encoder, creating a "super-encoder" that processes raw audio directly in the ONNX graph. This is faster then running a seperate process to extract the log-mel spectrogram and then passing this into the encoder.


**Decoder Optimization:** Employs both `decoder.onnx` and `decoder_with_past.onnx` for efficient autoregressive generation with KV-cache.

### Asset Generation

Run from the project root (requires Python venv setup in `conversion_tooling/`):

Make sure you have a python environment with the required dependencies:
```
cd conversion_tooling
python -m venv venv
source venv/bin/activate
pip install -r requirements
```

then run the asset generation
```
cd ..
./build_assets.sh
```

This script:
1. Downloads Whisper models from HuggingFace using `convert_whisper_to_onnx.py`
2. Generates preprocessor with `export_whisper_preprocessor.py`
3. Merges preprocessor + encoder into super-encoder using `merge_preprocessor_encoder.py`
4. Outputs to `assets/transcribers/whisper/models/{default,default_int8,default_int8_optimum}/`

## Testing

**Unit Tests** (`test/`):
- `audio_test.dart`: Audio loading utilities
- `whisper_tokenizer_test.dart`: Tokenizer encoding/decoding
- `whisper_test.dart`: Whisper transcriber functionality
- `whisper_streaming_test.dart`: Streaming transcriber functionality

Run with: `flutter test`

**Integration Tests** (`integration_test/`):

Run with: `flutter test integration_test/`

**Demo Applications** (`example/lib/`):

This library provides demo apps to demonstrate library usage:
- `app_nonstreaming.dart`: Record and transcribe complete audio files
- `app_streaming.dart`: Real-time streaming transcription with live partial results

## Performance Measurement

Measured on test audio (`assets/audio/jfk_asknot.wav`, 11 seconds) in non-streaming mode. Run `flutter test integration_test/whisper_test.dart` (from `example/` directory) to measure on your device.

| Device | Model | Inference time (avg ± std) |
| -- | -- | -- |
| Macbook Pro M2 | default | 576.4 ± 37.3 ms |
| Macbook Pro M2 | default_int8 | 477.8 ± 22.1 ms |
| Samsung Galaxy 11A+ Tablet | default | 7758.6 ± 121.4 ms |
| Samsung Galaxy 11A+ Tablet | default_int8 | 1137.0 ± 34.3 ms |
| Huawei Y9 Prime 2019 (STK-L21) | default_int8 | 3370.0 +- 120.5 ms|
| Samsung Tablet SM X115 | default_int8 | 1509.8 +- 48.9 ms|



**Quantization Impact:**
- Mac M4: int8 provides ~1.2x speedup (minimal benefit)
- Samsung Tablet: int8 provides ~6.8x speedup (significant benefit)

The int8 quantization is much more effective on mobile devices with less optimized hardware for full-precision inference.

Future work: Extend measurements to corpus with varying audio lengths.


## Installation


### Setting Up This Project for Development

### Clone the repository and install dependencies
   ```bash
   git clone <repository-url>
   cd flutter_ondevice_asr
   flutter pub get
   ```

### Generate assets

The library requires Whisper model files and tokenizer vocabularies. Generate them by running:
```bash
./build_assets.sh
```

This requires a Python virtual environment in `conversion_tooling/` (see Asset Generation section above).


### Running Unit Tests

Unit tests (`flutter test`) run in a pure Dart VM without building native libraries. Since this library uses `onnxruntime_v2` (which uses FFI to call native ONNX Runtime), you need to install the native library locally for unit tests to work.

Integration tests (`integration_test/`) don't require this setup as they build the full app with native libraries included automatically.

### macOS Setup

1. **Install ONNX Runtime via Homebrew:**
   ```bash
   brew install onnxruntime
   ```

   This installs ONNX Runtime 1.23.2+ (compatible with `onnxruntime_v2: ^1.23.2`).

2. **Create version compatibility symlink:**

   The package looks for `libonnxruntime.1.21.0.dylib`, but Homebrew installs version 1.23.2. Create a compatibility symlink:
   ```bash
   cd /opt/homebrew/Cellar/onnxruntime/1.23.2_2/lib
   ln -s libonnxruntime.1.23.2.dylib libonnxruntime.1.21.0.dylib
   ```

   Note: Adjust the version path (`1.23.2_2`) if Homebrew installed a different version. Check with: `ls /opt/homebrew/Cellar/onnxruntime/`

3. **Create symlink in project root:**
   ```bash
   cd /path/to/flutter_onnx_whisper
   ln -sf /opt/homebrew/Cellar/onnxruntime/1.23.2_2/lib/libonnxruntime.1.21.0.dylib .
   ```


# Licence

TODO

Also add licence of included libraries, whisper models, whisper tokenizer and silero vad