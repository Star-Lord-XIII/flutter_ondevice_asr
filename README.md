# Flutter On-Device ASR

A Flutter library for on-device automatic speech recognition (ASR):
- Model-agnostic architecture supporting arbitrary ASR models
- Streaming and non-streaming transcription
- ONNX Runtime-based inference (different quantization schemes)

The Whisper transcriber implementation has been tested on several different Android devices. When based on Whisper-tiny, it should run on mid-level phones in streaming mode. A fallback to non-streaming or semi-streaming (wait until segment end detected) should allow weaker devices to handle on-device transcription as well.

## Library Structure

This library uses a **model-agnostic architecture** that separates transcription models from streaming logic, making it easy to support multiple ASR models with a unified API.

**Core Abstractions:**
- `Transcriber`: Abstract base interface for all ASR models (defines `loadModels()`, `transcribe()`, `transcribeFile()`)
- `StreamingTranscriber`: Model-agnostic streaming implementation that works with any `Transcriber`
- `TranscriptionResult`: Unified result format for both streaming and non-streaming transcription, containing:
  - Transcribed text
  - Final/partial indicator (for streaming)
  - Audio duration and timestamp
  - Optional word-level confidence scores with average confidence calculation
- `OnnxConfig`: Runtime configuration for ONNX models

**Model Implementations:**
- `WhisperTranscriber`: Whisper model implementation in `lib/models/whisper/` (implements `Transcriber`)
- Additional models can be added by implementing the `Transcriber` interface

**Shared Components (used by all models):**
- `Audio`: Audio loading and preprocessing utilities
- `SileroVAD`: Voice activity detection for streaming
- Streaming logic: VAD-based segmentation and partial transcription

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

### Combined Preprocessing and Encoding

#### ONNX-Native Preprocessing

All audio preprocessing (mel spectrogram extraction) is done within ONNX Runtime using standard operators (HannWindow, STFT, MatMul, Log, Pad). The preprocessor is defined in Python using `onnxscript` (`conversion_tooling/whisper_preprocessor.py`) and exported to ONNX format, ensuring it matches the core Whisper implementation exactly. This approach offers several benefits over Dart-based FFT:
- **Performance**: ~15% faster (160ms → 135ms measured on macOS), with bigger gains expected on Android devices due to NEON/SIMD optimizations
- **Quality**: Better transcription accuracy by using the exact same preprocessing operations as during Whisper training
- **Efficiency**: Fewer memory allocations and reduced GC pressure by keeping data in native memory
- **Portability**: Same ONNX graph works across all platforms

This requires ONNX Runtime 1.22+ (opset 17 support), which is why we use `onnxruntime_v2`.

#### Super-Encoder

The preprocessing stage (log-mel spectrogram conversion) is merged into the encoder, creating a "super-encoder" that processes raw audio directly in the ONNX graph. This is faster then running a seperate process to extract the log-mel spectrogram and then passing this into the encoder.

By merging the preprocessing and encoder into one onnx model/graph, we eliminate the need to transfer intermediate tensors between the host language (here: Dart) and the inference runtime (her: Onnx). This should be particularly valuable on mobile devices where memory bandwidth is limited and minimizing data transfers between different execution contexts significantly impacts performance and battery life.

It also makes the code much simpler and abstracts preprocessing logic from the app (better guarantee that preprocessing during inference is the same as during training, which is critical for performance as encoder is very sensitive to preprocessing changes).

### Decoder

* We both `decoder.onnx` and `decoder_with_past.onnx` for efficient autoregressive generation with KV-cache. 
* The FFI (Foreign Function Interface) communication in `onnxruntime_v2` enables direct Dart-to-native function calls with minimal overhead, avoiding the serialization cost (serializing/deserializing data) of MethodChannel alternatives (eg flutter_onnxruntime). This is essential for autoregressive decoding, which makes many decoder calls per transcription (depending on output length), where MethodChannel overhead would add overhead of wasted time.

### Asset Generation

Right now, all model specific asset files for the Whisper transcriber can be generated.

This will first download the multilingual Whisper tiny model from HuggingFace, convert to Onnx using Optimum, then create the preprocessor as seperate Onnx model and eventually merge that with the encoder into a "super encoder" onnx model. All files are then moved to the assets folder.

We create both the f16 as well as the int8 qunatized version. For usage, unless quality impacts are too significant, it is highly recommended to use the int8 version.


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

The library requires Whisper model files and tokenizer vocabularies. You need to generate them with the `conversion_tooling/`.

Create a python environment and the required dependencies:
```
cd conversion_tooling
python -m venv venv
source venv/bin/activate
pip install -r requirements
```

Then run the asset generation script
```
cd ..
./build_assets.sh
```


This script runs through these steps:

1. Downloads Whisper models from HuggingFace using `convert_whisper_to_onnx.py`
2. Generates preprocessor with `export_whisper_preprocessor.py`
3. Merges preprocessor + encoder into super-encoder using `merge_preprocessor_encoder.py`
4. Outputs to `assets/transcribers/whisper/models/{default,default_int8,default_int8_optimum}/`


### Running Unit Tests

Unit tests (`flutter test`) run in a pure Dart VM without building native libraries. Since this library uses `onnxruntime_v2` (which uses FFI to call native ONNX Runtime), you need to install the native library locally for unit tests to work.

The unit tests are on purpose maximally verbose. For the transcription and especially the streaming transcription test, this will show information for every 512 frame sample being recorded and allows to track the streaming decisions for debugging purposes.

This level of verbosity needs to be avoided in a production setting!

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