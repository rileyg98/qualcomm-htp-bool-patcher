# HTP BOOL Constraint Enforcer

A robust utility to modify ONNX models for compatibility with the Qualcomm Hexagon NPU (HTP backend). It ensures that BOOL tensors are only used in supported operations (per the [QNN Operation Definition Supplement](https://docs.qualcomm.com/doc/80-63442-10/topic/HtpOpDefSupplement.html)) by injecting `Cast` nodes (BOOL ↔ INT32).

## Features

- **Spec-Driven**: Uses a JSON allowlist generated from official QNN documentation.
- **Sandwich Algorithm**: Automatically injects Pre-Casts (BOOL → INT32) for unsupported inputs and Post-Casts (INT32 → BOOL) to maintain graph invariants.
- **Initializer Support**: Converts BOOL Initializers directly to INT32 to minimize graph overhead.
- **Verification**: Runs `onnx.checker` on the output model to ensure validity.

## Usage

### 1. Fix ONNX Model
```bash
python3 fix_bool_graph.py <input_model.onnx> qnn_bool8_allowlist.json --output <output_model.onnx>
```

## Example
```bash
python3 fix_bool_graph.py model_with_bools.onnx qnn_bool8_allowlist.json --output model_htp_compliant.onnx
```

