#!/bin/bash

# Script to download Whisper tokenizer vocab files from HuggingFace
#
# Vocab files are centralized in assets/tokenizer/:
# - vocab_en.json: For English-only models (tiny.en, base.en, small.en, etc.)
# - vocab_multilingual.json: For multilingual models (tiny, base, small, etc.)
#
# Each model directory only needs generation_config.json (model-specific token IDs)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKENIZER_DIR="$SCRIPT_DIR/assets/tokenizer"

echo "Downloading Whisper tokenizer vocab files..."
echo "Target directory: $TOKENIZER_DIR"

# Create tokenizer directory if it doesn't exist
mkdir -p "$TOKENIZER_DIR"

# Download multilingual vocab (for whisper-tiny, whisper-small, etc.)
echo ""
echo "Downloading multilingual vocab..."
curl -L "https://huggingface.co/openai/whisper-tiny/raw/main/vocab.json" \
  -o "$TOKENIZER_DIR/vocab_multilingual.json"
echo "✓ Downloaded vocab_multilingual.json ($(du -h "$TOKENIZER_DIR/vocab_multilingual.json" | cut -f1))"

# Download English-only vocab (for whisper-tiny.en, whisper-small.en, etc.)
echo ""
echo "Downloading English-only vocab..."
curl -L "https://huggingface.co/openai/whisper-tiny.en/raw/main/vocab.json" \
  -o "$TOKENIZER_DIR/vocab_en.json"
echo "✓ Downloaded vocab_en.json ($(du -h "$TOKENIZER_DIR/vocab_en.json" | cut -f1))"

echo ""
echo "Done! All vocab files downloaded successfully."
echo ""
echo "Files in $TOKENIZER_DIR:"
ls -lh "$TOKENIZER_DIR"/*.json
