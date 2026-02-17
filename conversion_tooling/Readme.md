# Py tooling for conversion of HF whisper models to onnx

* can be used for testing here
* later needs to be integrated into training pipeline

## ONNX Whisper processor
* based on https://github.com/istupakov/onnx-asr
    * onnx-asr preprocessor implementation
    * this class defines the preprocessor to be exported to onnx: conversion_tooling/whisper_preprocessor.py
    * that is MIT licence so copying should be fine, but must mention licence and copy original licence text

* procedure
    * run `python export_whisper_preprocessor.py`
    * then copy created asset: `cp whisper_preprocessor_80.onnx ../assets/models/shared/`
