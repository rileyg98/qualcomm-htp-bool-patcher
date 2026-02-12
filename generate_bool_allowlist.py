import json
import re
import sys
import os

# Mapping from QNN Op names (in spec) to ONNX Op types
OP_NAME_MAPPING = {
    'ElementWiseAnd': 'And',
    'ElementWiseOr': 'Or',
    'ElementWiseXor': 'Xor',
    'ElementWiseNot': 'Not',
    'ElementWiseEqual': 'Equal',
    'ElementWiseGreater': 'Greater',
    'ElementWiseLess': 'Less',
    'ElementWiseGreaterEqual': 'GreaterOrEqual',
    'ElementWiseLessEqual': 'LessOrEqual',
    'ElementWiseSelect': 'Where',
    'ElementWiseNotEqual': 'NotEqual',
    'ElementWiseUnary': 'Not', 
    'Reshape': 'Reshape',
    'Transpose': 'Transpose',
    'Split': 'Split',
    'Concat': 'Concat',
    'Tile': 'Tile',
    'Cast': 'Cast',
    'Slice': 'Slice',
    'Squeeze': 'Squeeze',
    'Unsqueeze': 'Unsqueeze',
    'Gather': 'Gather',
    'ScatterNd': 'ScatterND', 
    'Gru': 'GRU',
    'Lstm': 'LSTM',
    'IsInf': 'IsInf',
    'IsNan': 'IsNaN',
    'NonZero': 'NonZero',
    'RoiAlign': 'RoiAlign',
}

def get_onnx_op_type(qnn_op_name):
    return OP_NAME_MAPPING.get(qnn_op_name, qnn_op_name.strip())

def parse_bool_support_spec(spec_path, output_json_path):
    allowed_ops = {}
    
    try:
        with open(spec_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Spec file not found at {spec_path}")
        sys.exit(1)

    table_started = False
    
    for line in lines:
        line = line.strip()
        if not line.startswith('|'):
            continue
        if '---' in line:
            table_started = True
            continue
            
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) < 2:
            continue
            
        qnn_op_name = parts[0].replace('*', '').strip() 
        params_str = parts[1]
        
        onnx_op_type = get_onnx_op_type(qnn_op_name)
        
        if onnx_op_type not in allowed_ops:
            allowed_ops[onnx_op_type] = {'inputs': [], 'outputs': []}
            
        # Parse inputs
        inputs_matches = re.finditer(r'in\[(\d+)(?:\.\.([a-z]+))?\]', params_str)
        for m in inputs_matches:
            start_idx = int(m.group(1))
            end_marker = m.group(2)
            if end_marker:
                # Expand range
                allowed_ops[onnx_op_type]['inputs'].extend(list(range(start_idx, 64)))
            else:
                allowed_ops[onnx_op_type]['inputs'].append(start_idx)
                
        # Parse outputs
        outputs_matches = re.finditer(r'out\[(\d+)(?:\.\.([a-z]+))?\]', params_str)
        for m in outputs_matches:
            start_idx = int(m.group(1))
            end_marker = m.group(2)
            if end_marker:
                 allowed_ops[onnx_op_type]['outputs'].extend(list(range(start_idx, 64)))
            else:
                allowed_ops[onnx_op_type]['outputs'].append(start_idx)
    
    # Sort and uniques
    for op in allowed_ops:
        allowed_ops[op]['inputs'] = sorted(list(set(allowed_ops[op]['inputs'])))
        allowed_ops[op]['outputs'] = sorted(list(set(allowed_ops[op]['outputs'])))

    with open(output_json_path, 'w') as f:
        json.dump(allowed_ops, f, indent=2)
        
    print(f"Generated {output_json_path} with {len(allowed_ops)} ops.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 generate_bool_allowlist.py <spec_md_file> [output_json_file]")
        sys.exit(1)
        
    spec_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "qnn_bool8_allowlist.json"
    
    parse_bool_support_spec(spec_path, output_path)
