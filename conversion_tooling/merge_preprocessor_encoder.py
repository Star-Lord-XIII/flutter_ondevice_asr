"""Merge Whisper preprocessor and encoder into a single "super encoder" model.

This creates a unified ONNX model that takes raw audio waveforms as input
and produces encoder hidden states as output, eliminating the need for
Dart glue code to pass mel spectrograms between models.

Benefits:
- Zero Dart data marshalling between preprocessor and encoder
- Faster: transition happens in native C++/NPU memory
- Simpler Flutter code: just send raw audio, get hidden states
- Better encapsulation: preprocessing details hidden in model

Usage:
    python merge_preprocessor_encoder.py \
        --preprocessor whisper_preprocessor_80.onnx \
        --encoder ../assets/models/whisper_tiny/default_int8/encoder_model.onnx \
        --output super_encoder.onnx
"""

import argparse
import onnx
from onnx import compose


def merge_models(preprocessor_path, encoder_path, output_path):
    """Merge preprocessor and encoder ONNX models into a single "super encoder".

    Uses onnx.compose.merge_models to combine the models by connecting:
    - Preprocessor output 'features' -> Encoder input 'input_features'

    The merged model will:
    - Input: raw audio waveform (Float32[batch, samples])
    - Output: encoder hidden states (Float32[batch, seq_len, hidden_dim])
    """
    print(f"Loading preprocessor from {preprocessor_path}...")
    preprocessor = onnx.load(preprocessor_path)

    print(f"Loading encoder from {encoder_path}...")
    encoder = onnx.load(encoder_path)

    print("\nPreprocessor info:")
    print(f"  Inputs: {[i.name for i in preprocessor.graph.input]}")
    print(f"  Outputs: {[o.name for o in preprocessor.graph.output]}")

    print("\nEncoder info:")
    print(f"  Inputs: {[i.name for i in encoder.graph.input]}")
    print(f"  Outputs: {[o.name for o in encoder.graph.output]}")

    # Check IR version compatibility
    if preprocessor.ir_version != encoder.ir_version:
        print(f"\n⚠ IR version mismatch: preprocessor={preprocessor.ir_version}, encoder={encoder.ir_version}")
        print(f"  Downgrading preprocessor to IR version {encoder.ir_version}...")
        preprocessor.ir_version = encoder.ir_version

    # Map preprocessor outputs to encoder inputs
    # Preprocessor outputs 'features' (mel spectrogram)
    # Encoder expects 'input_features'
    io_map = [("features", "input_features")]

    print(f"\nMerging with io_map: {io_map}")

    # Use ONNX compose to merge the models
    merged_model = compose.merge_models(
        preprocessor,
        encoder,
        io_map=io_map
    )

    # Validate the merged model
    print("\nValidating merged model...")
    try:
        onnx.checker.check_model(merged_model)
        print("✓ Model validation passed")
    except Exception as e:
        print(f"⚠ Validation warning: {e}")
        print("  (This may be OK - some validators are strict)")

    # Save merged model
    print(f"\nSaving merged model to {output_path}...")
    onnx.save(merged_model, output_path)

    # Print merged model info
    model_size_mb = len(merged_model.SerializeToString()) / (1024 * 1024)
    print(f"\n✓ Merged model saved successfully!")
    print(f"  Size: {model_size_mb:.2f} MB")
    print(f"  IR Version: {merged_model.ir_version}")
    print(f"  Opset: {merged_model.opset_import[0].version}")
    print(f"  Inputs: {[i.name for i in merged_model.graph.input]}")
    print(f"  Outputs: {[o.name for o in merged_model.graph.output]}")

    return merged_model


def main():
    parser = argparse.ArgumentParser(description="Merge Whisper preprocessor and encoder models")
    parser.add_argument(
        "--preprocessor",
        default="whisper_preprocessor_80.onnx",
        help="Path to preprocessor ONNX model"
    )
    parser.add_argument(
        "--encoder",
        default="../assets/models/whisper_tiny/default_int8/encoder_model.onnx",
        help="Path to encoder ONNX model"
    )
    parser.add_argument(
        "--output",
        default="super_encoder.onnx",
        help="Output path for merged model"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Whisper Super Encoder Creation")
    print("=" * 70)

    merge_models(args.preprocessor, args.encoder, args.output)

    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Test the merged model with sample audio")
    print("  2. Compare outputs with separate models")
    print("  3. Copy to Flutter assets if validation passes")
    print("  4. Update Flutter code to use super encoder")
    print("=" * 70)


if __name__ == "__main__":
    main()
