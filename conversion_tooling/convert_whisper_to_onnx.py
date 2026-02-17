# straight forward conversion a HF Transformer Whisper model to ONNX with Optimum
# creates full prec and int8 quant
# example
# python convert_whisper_to_onnx.py "openai/whisper-tiny" /tmp/onnx_tiny


from optimum.onnxruntime import ORTModelForSpeechSeq2Seq, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from onnxruntime import GraphOptimizationLevel
from onnxruntime.tools.optimize_onnx_model import optimize_model
from onnxruntime.quantization import QuantType, quantize_dynamic
import glob
from pathlib import Path

from transformers import WhisperProcessor
import os, shutil
import time
import argparse
import shutil


def convert_to_onnx(original_model_path, default_onnx_folder):
    # base conversion to ONNX (uses default opset from Optimum library)
    # Note: opset parameter not supported in Optimum 2.1.0
    ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(
        original_model_path,
        export=True,
    )
    ort_model.save_pretrained(default_onnx_folder)
    print(f"Model saved to {default_onnx_folder}")

def copy_config_files(source_dir, target_dir):

    for json_file in glob.glob(os.path.join(source_dir, "*.json")):
        shutil.copy2(json_file, target_dir)


def quantize_onnx(onnx_input_dir, quantized_onnx_output_dir):
    # quantize to int8
    # one could quantize the optimized version as well but I didn't see much improvements here    
    print(f"\n>>> Quantizing models from {onnx_input_dir}...")

    onnx_model_paths = glob.glob(os.path.join(onnx_input_dir, "*.onnx"))
    for model_path in onnx_model_paths:
        base_name = os.path.basename(model_path)
        output_path_quantized = os.path.join(quantized_onnx_output_dir, base_name)
        print(f">>> Quantizing {base_name}...")
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path_quantized,
            # only do MatMul quantization (if doing conv, inference fails)
            op_types_to_quantize=['MatMul'],
            weight_type=QuantType.QInt8,
            extra_options={'DefaultTensorType': 1}  # 1 = FLOAT
        )
    copy_config_files(onnx_input_dir, quantized_onnx_output_dir)
    print(f"Quantized {onnx_input_dir}.")


# Note: this would technically be the most optimized version to quantize whisper models with special optimum
# exporter quant settings. While the model gets smaller, it is not better or faster.
# see details here: https://linear.app/cdli/issue/EUPHAPP-99/support-quantized-models-for-faster-on-device-inference
# can improve later but for now keeping standard quantization
def quantize_onnx_refined(onnx_input_dir, quantized_onnx_output_dir):
    print(f"\n>>> Quantizing models from {onnx_input_dir}...")
    
    onnx_files = glob.glob(os.path.join(onnx_input_dir, "*.onnx"))
    for model_path in onnx_files:
        file_name = os.path.basename(model_path)
        print(f">>> Processing {file_name}...")
        quantizer = ORTQuantizer.from_pretrained(onnx_input_dir, file_name=file_name)

    
        # arm for android and apple silicon
        # This config optimizes the math for the ARM NEON instruction set
        dqconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=True)
        dqconfig.operators_to_quantize = [
            op for op in dqconfig.operators_to_quantize 
            if "Conv" not in op
        ]
        
        quantizer.quantize(
            save_dir=quantized_onnx_output_dir,
            quantization_config=dqconfig,
            file_suffix="quantized" # This avoids overwriting issues
        )

    # Clean up filenames (Optimum adds a suffix, let's rename them back)
    # This ensures they match the filenames expected by the generation config
    for quant_file in glob.glob(os.path.join(quantized_onnx_output_dir, "*_quantized.onnx")):
        new_name = quant_file.replace("_quantized.onnx", ".onnx")
        os.rename(quant_file, new_name)

    copy_config_files(onnx_input_dir, quantized_onnx_output_dir)

def main(original_model_path, onnx_output_folder):
    """
    Convert a HuggingFace Whisper model to ONNX format with optimization and quantization.

    Args:
        original_model_path: Path to the Whisper model (HuggingFace Hub ID or local path)
        onnx_output_folder: Output folder for ONNX models
    """
    default_onnx_folder = os.path.join(onnx_output_folder, 'default')
    default_int8_onnx_folder = os.path.join(onnx_output_folder, 'default_int8')
    default_int8_onnx_folder_v2 = os.path.join(onnx_output_folder, 'default_int8_optimum')

    os.makedirs(default_onnx_folder, exist_ok=True)
    os.makedirs(default_int8_onnx_folder, exist_ok=True)

    # Convert to ONNX
    convert_to_onnx(original_model_path, default_onnx_folder)

    # Quantize both default and optimized versions
    quantize_onnx(default_onnx_folder, default_int8_onnx_folder)

    # new quantize method
    quantize_onnx_refined(default_onnx_folder, default_int8_onnx_folder_v2)

    print(f"\n✓ Conversion complete! Models saved to {onnx_output_folder}")


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Convert Whisper model to ONNX format')
    parser.add_argument('original_model_path', type=str, default="openai/whisper-tiny", help='Path to the Whisper model (HuggingFace Hub ID or local path)')
    parser.add_argument('onnx_output_folder', type=str, help='Base folder where onnx model will be written to (each in subfolders)')
    args = parser.parse_args()

    main(args.original_model_path, args.onnx_output_folder)    