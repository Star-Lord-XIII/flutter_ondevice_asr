#!/bin/bash
# Build optimized assets for Flutter bundle from HuggingFace models
#
# This script:
# 1. Exports Whisper models from HuggingFace to models/ (temp directory)
# 2. Exports preprocessor to models/
# 3. Merges preprocessor + encoder into super_encoder.onnx for each variant
# 4. Copies super_encoder, decoders, and configs to assets/
# 5. Removes standalone encoders from assets/
#
# Usage:
#   ./build_assets.sh [model_id]
#
# Examples:
#   ./build_assets.sh                      # Uses default: openai/whisper-tiny
#   ./build_assets.sh openai/whisper-base  # Use a different model

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"
CONVERSION_DIR="$PROJECT_ROOT/conversion_tooling"
MODELS_DIR="$PROJECT_ROOT/models"  # Temporary build directory (gitignored)
ASSETS_DIR="$PROJECT_ROOT/assets/transcribers/whisper/models"

# Default model ID (can be overridden)
MODEL_ID="${1:-openai/whisper-tiny}"

echo "======================================================================="
echo "Building Flutter Assets from HuggingFace"
echo "======================================================================="
echo "Model: $MODEL_ID"
echo "Temp directory: $MODELS_DIR"
echo "Output: $ASSETS_DIR"
echo ""

# Check virtual environment
if [ ! -d "$CONVERSION_DIR/venv" ]; then
    echo "❌ Error: Virtual environment not found."
    echo "Run: cd conversion_tooling && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source "$CONVERSION_DIR/venv/bin/activate"

# Clean and create temp models directory
echo "1. Creating temporary models directory..."
rm -rf "$MODELS_DIR"
mkdir -p "$MODELS_DIR/whisper_tiny"
mkdir -p "$MODELS_DIR/preprocessor"

# Step 1: Export Whisper models from HuggingFace
echo ""
echo "======================================================================="
echo "Step 1: Exporting Whisper models from HuggingFace"
echo "======================================================================="
echo "This will create 3 variants: default, default_int8, default_int8_optimum"
echo ""

python "$CONVERSION_DIR/convert_whisper_to_onnx.py" \
    "$MODEL_ID" \
    "$MODELS_DIR/whisper_tiny"

echo ""
echo "✓ Whisper models exported successfully!"

# Step 2: Export preprocessor
echo ""
echo "======================================================================="
echo "Step 2: Exporting preprocessor"
echo "======================================================================="

cd "$CONVERSION_DIR"
python export_whisper_preprocessor.py

# Move to models directory
mv whisper_preprocessor_80.onnx "$MODELS_DIR/preprocessor/"

echo ""
echo "✓ Preprocessor exported successfully!"

# Step 3: Build assets for each variant
echo ""
echo "======================================================================="
echo "Step 3: Building optimized assets"
echo "======================================================================="

# Function to build assets for one model variant
build_variant() {
    local variant=$1
    echo ""
    echo "-------------------------------------------------------------------"
    echo "Building: whisper_tiny/$variant"
    echo "-------------------------------------------------------------------"

    local source_dir="$MODELS_DIR/whisper_tiny/$variant"
    local target_dir="$ASSETS_DIR/whisper_tiny/$variant"

    # Check source directory exists
    if [ ! -d "$source_dir" ]; then
        echo "⚠️  Skipping $variant - source directory not found"
        return
    fi

    # Check encoder exists
    if [ ! -f "$source_dir/encoder_model.onnx" ]; then
        echo "⚠️  Skipping $variant - encoder_model.onnx not found"
        return
    fi

    echo "  → Creating super encoder (preprocessor + encoder)..."
    python "$CONVERSION_DIR/merge_preprocessor_encoder.py" \
        --preprocessor "$MODELS_DIR/preprocessor/whisper_preprocessor_80.onnx" \
        --encoder "$source_dir/encoder_model.onnx" \
        --output "$CONVERSION_DIR/super_encoder_temp.onnx" \
        > /dev/null

    echo "  → Creating target directory..."
    mkdir -p "$target_dir"

    echo "  → Copying super encoder to assets..."
    cp "$CONVERSION_DIR/super_encoder_temp.onnx" "$target_dir/super_encoder.onnx"

    echo "  → Copying decoders to assets..."
    cp "$source_dir/decoder_model.onnx" "$target_dir/"
    cp "$source_dir/decoder_with_past_model.onnx" "$target_dir/"

    echo "  → Copying config files to assets..."
    cp "$source_dir/config.json" "$target_dir/"
    cp "$source_dir/generation_config.json" "$target_dir/"

    echo "  → Removing standalone encoder from assets (if exists)..."
    rm -f "$target_dir/encoder_model.onnx"

    echo "  → Cleaning up temporary files..."
    rm -f "$CONVERSION_DIR/super_encoder_temp.onnx"

    echo ""
    echo "  ✓ Built $variant successfully!"
    ls -lh "$target_dir" | grep -E "(super_encoder|decoder|config)" | awk '{print "      " $9 " (" $5 ")"}'
}

# Build all three variants
build_variant "default"
build_variant "default_int8"
build_variant "default_int8_optimum"

echo ""
echo "======================================================================="
echo "✓ All assets built successfully!"
echo "======================================================================="
echo ""
echo "Summary:"
echo "  - Exported models from: $MODEL_ID"
echo "  - Created super_encoder.onnx for each variant (preprocessor + encoder merged)"
echo "  - Copied decoders and configs to assets/"
echo "  - Removed standalone encoders from bundle"
echo ""
echo "Temporary files:"
echo "  - Source models: $MODELS_DIR (can be deleted or kept for caching)"
echo ""
echo "Bundled assets (shipped with app):"
echo "  - $ASSETS_DIR/"