"""Setup for Whisper ONNX Conversion Tooling.

This package provides tools for converting Whisper models to ONNX format
with optimizations for on-device inference (mobile/embedded).

Includes:
- HuggingFace Whisper to ONNX conversion
- INT8 quantization
- ONNX-native preprocessor generation (based on onnx-asr)
- Preprocessor + encoder merging (super-encoder)

Model input can be:
- HuggingFace model ID (e.g., "openai/whisper-tiny")
- Local path to fine-tuned Whisper model

Install from Git:
    pip install git+https://github.com/your-org/flutter_ondevice_asr.git#subdirectory=conversion_tooling

Usage:
    from convert_whisper_to_onnx import run_conversion
    run_conversion("openai/whisper-tiny", "./output")
"""

from setuptools import setup

setup(
    name="whisper-conversion-tooling",
    version="0.1.0",
    description="Whisper model conversion to optimized ONNX for on-device inference",
    long_description=__doc__,
    long_description_content_type="text/plain",
    author="Katrin Tomanek",
    url="https://github.com/your-org/flutter_ondevice_asr",
    py_modules=[
        "whisper_preprocessor",
        "convert_whisper_to_onnx",
    ],
    install_requires=[
        "onnx>=1.15.0",
        "onnxscript>=0.1.0",
        "torchaudio>=2.0.0",
        "optimum[onnxruntime]>=1.16.0",
        "transformers>=4.30.0",
        "torch>=2.0.0",
    ],
    python_requires=">=3.9",
)
