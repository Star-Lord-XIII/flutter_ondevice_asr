"""Export Whisper preprocessor to ONNX format.

This script exports the ONNX-native Whisper mel spectrogram preprocessor
to a .onnx file that can be used in the Flutter app.

Whisper uses 80 mel bands (not 128), so only the 80-band version is exported.

Usage:
    python export_whisper_preprocessor.py

This will create whisper_preprocessor_80.onnx in the current directory.
"""

import onnx
from whisper_preprocessor import WhisperPreprocessor80


def export_preprocessor_80():
    """Export 80-band mel spectrogram preprocessor for Whisper."""
    print("Exporting WhisperPreprocessor80...")
    model = WhisperPreprocessor80.to_model_proto()

    # Set IR version to 8 to match encoder (encoder uses IR 8, onnxscript creates IR 10)
    model.ir_version = 8

    onnx.save(model, "whisper_preprocessor_80.onnx")
    print(f"✓ Saved to whisper_preprocessor_80.onnx (IR version: {model.ir_version}, Opset: {model.opset_import[0].version})")
    print("\nNext steps:")
    print("  cp whisper_preprocessor_80.onnx ../assets/models/shared/")


if __name__ == "__main__":
    export_preprocessor_80()
    print("\n✓ Preprocessor exported successfully!")
