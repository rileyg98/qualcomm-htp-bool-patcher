import onnx
import onnx.checker
import onnx.shape_inference
import sys
import argparse
import json
import logging
from onnx import TensorProto, helper
from collections import defaultdict, deque

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def topological_sort(graph):
    """
    Sorts nodes topologically using Kahn's Algorithm.
    Ensures final graph is properly ordered for HTP backend.
    """
    logger.info(">>> Sorting nodes topologically...")
    
    # Identify initial available tensors (inputs and initializers)
    available_tensors = set(i.name for i in graph.input)
    available_tensors.update(i.name for i in graph.initializer)
    
    # Map every tensor name to its producer (node index)
    producer_map = {}
    for i, node in enumerate(graph.node):
        for output_name in node.output:
            if output_name:
                producer_map[output_name] = i
                
    # Build adjacency list (node -> consumers) and calculate in-degrees
    node_dependencies = defaultdict(list)
    node_in_degree = defaultdict(int)
    
    for i, node in enumerate(graph.node):
        for input_name in node.input:
            if input_name in producer_map:
                producer_idx = producer_map[input_name]
                node_dependencies[producer_idx].append(i)
                node_in_degree[i] += 1
                
    # Queue of nodes with 0 dependencies on other nodes
    queue = deque([i for i in range(len(graph.node)) if node_in_degree[i] == 0])
    sorted_nodes = []
    
    while queue:
        node_idx = queue.popleft()
        sorted_nodes.append(graph.node[node_idx])
        
        if node_idx in node_dependencies:
            for child_idx in node_dependencies[node_idx]:
                node_in_degree[child_idx] -= 1
                if node_in_degree[child_idx] == 0:
                    queue.append(child_idx)
                    
    if len(sorted_nodes) != len(graph.node):
        logger.error(f"❌ Cycle detected or graph disconnected! Sort failed ({len(sorted_nodes)}/{len(graph.node)} nodes sorted).")
        return False
        
    logger.info("✅ Graph successfully sorted.")
    del graph.node[:]
    graph.node.extend(sorted_nodes)
    return True

def load_bool_support_spec(json_path):
    """
    Loads the JSON allowlist of (OpType, InputIndex).
    Returns a dict: { OpType: { 'inputs': set(indices), 'outputs': set(indices) } }
    """
    try:
        with open(json_path, 'r') as f:
            allowed_ops_raw = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: Spec file not found at {json_path}")
        sys.exit(1)
        
    allowed_ops = {}
    for op, data in allowed_ops_raw.items():
        allowed_ops[op] = {
            'inputs': set(data.get('inputs', [])),
            'outputs': set(data.get('outputs', []))
        }
    return allowed_ops

def get_tensor_type(value_info):
    """Returns the DataType of a value_info proto."""
    if not value_info.HasField('type'):
        return None
    if not value_info.type.HasField('tensor_type'):
        return None
    return value_info.type.tensor_type.elem_type

def make_cast_node(input_name, output_name, to_type, name=None):
    """Creates a Cast node."""
    node = helper.make_node(
        'Cast',
        inputs=[input_name],
        outputs=[output_name],
        to=to_type
    )
    if name:
        node.name = name
    return node

def enforce_htp_bool_constraints(model_path, spec_path, output_path):
    logger.info(f"Loading model from {model_path}...")
    model = onnx.load(model_path)
    
    # 0. Global Shape Inference
    logger.info("Running global shape inference...")
    try:
        inferred_model = onnx.shape_inference.infer_shapes(model)
    except Exception as e:
        logger.warning(f"Warning: Shape inference failed: {e}. Proceeding with partial inference info.")
        inferred_model = model
        
    value_types = {}
    for info in inferred_model.graph.value_info:
        value_types[info.name] = get_tensor_type(info)
        # Also store shape? No, just type for now.
    for info in inferred_model.graph.input:
        value_types[info.name] = get_tensor_type(info)
    for info in inferred_model.graph.output:
        value_types[info.name] = get_tensor_type(info)
        
    # Load Spec
    logger.info(f"Loading spec from {spec_path}...")
    allowed_ops = load_bool_support_spec(spec_path)
    
    graph = model.graph
    new_nodes = []
    
    nodes_processed = 0
    modifications = 0
    
    def get_type(name):
        return value_types.get(name, None)
        
    def supports_bool_input(op_type, index):
        if op_type not in allowed_ops:
            return False
        return index in allowed_ops[op_type]['inputs']

    # Initializer map for fast lookup
    initializers = {init.name: init for init in graph.initializer}
    
    for node in graph.node:
        op_type = node.op_type
        
        # --- Input Verification (Pre-Cast) ---
        new_inputs = []
        for i, input_name in enumerate(node.input):
            if not input_name: 
                new_inputs.append(input_name)
                continue
                
            current_type = get_type(input_name)
            
            if current_type == TensorProto.BOOL:
                if not supports_bool_input(op_type, i):
                    # Violation Detected
                    
                    if input_name in initializers:
                        # Case A: Initializer
                        init = initializers[input_name]
                        if init.data_type == TensorProto.BOOL:
                            logger.info(f"[FIX] {op_type} '{node.name}': Input {i} ('{input_name}') is BOOL Initializer -> Converting to INT32")
                            
                            try:
                                import numpy as np
                                from onnx import numpy_helper
                                arr = numpy_helper.to_array(init)
                                arr_int32 = arr.astype(np.int32)
                                new_init = numpy_helper.from_array(arr_int32, name=input_name)
                                
                                graph.initializer.remove(init)
                                graph.initializer.append(new_init)
                                initializers[input_name] = new_init 
                                value_types[input_name] = TensorProto.INT32
                                
                            except ImportError:
                                logger.error("NumPy not found. Cannot convert initializer. Skipping.")
                            
                            new_inputs.append(input_name)
                        else:
                            new_inputs.append(input_name)
                            
                    else:
                        # Case B: Activation
                        logger.info(f"[FIX] {op_type} '{node.name}': Input {i} ('{input_name}') is BOOL -> Injected Cast to INT32")
                        
                        unique_id = f"{nodes_processed}_{i}"
                        cast_node_name = f"Cast_to_INT32_{node.name}_{i}"
                        cast_out_name = f"{input_name}_to_int32_{unique_id}"
                        
                        cast_node = make_cast_node(input_name, cast_out_name, TensorProto.INT32, name=cast_node_name)
                        new_nodes.append(cast_node)
                        new_inputs.append(cast_out_name)
                        modifications += 1
                else:
                    new_inputs.append(input_name)
            else:
                new_inputs.append(input_name)
                
        node.input[:] = new_inputs 
        new_nodes.append(node)
        
        # --- Output Verification (Post-Cast) ---
        for j, out_name in enumerate(node.output):
            original_type = get_type(out_name)
            
            if original_type == TensorProto.BOOL:
                output_supported_in_spec = False
                if op_type in allowed_ops:
                    if j in allowed_ops[op_type]['outputs']:
                        output_supported_in_spec = True
                
                # If original output was BOOL, but Op is not in allowlist (meaning we likely forced inputs to INT32),
                # we must assume Op now outputs INT32 and Cast Back.
                # Note: If Op IS in allowlist, we didn't force inputs (or allowed bool inputs), so output stays BOOL.
                
                if not output_supported_in_spec:
                    logger.info(f"[RESTORE] {op_type} '{node.name}': Output {j} ('{out_name}') was BOOL but Op invalid for BOOL -> Injected Cast back to BOOL")
                    
                    current_out_name_int32 = f"{out_name}_int32_intermediate"
                    node.output[j] = current_out_name_int32
                    
                    unique_id = f"{nodes_processed}_{j}"
                    cast_back_name = f"Cast_back_to_BOOL_{node.name}_{j}"
                    cast_back_node = make_cast_node(current_out_name_int32, out_name, TensorProto.BOOL, name=cast_back_name)
                    new_nodes.append(cast_back_node)
                    modifications += 1

        nodes_processed += 1

    # Replace graph nodes
    del graph.node[:]
    graph.node.extend(new_nodes)

    # Topological Sort
    topological_sort(graph)
    
    logger.info(f"Finished. Processed {nodes_processed} nodes. Made {modifications} modifications.")
    logger.info(f"Saving compliant model to {output_path}...")
    onnx.save(model, output_path)
    
    try:
        onnx.checker.check_model(output_path)
        logger.info("Model verification passed.")
    except Exception as e:
        logger.error(f"Model verification failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enforce HTP BOOL constraints on ONNX model.")
    parser.add_argument("model_path", help="Path to input .onnx model")
    parser.add_argument("spec_json_path", help="Path to QNN allowlist JSON")
    parser.add_argument("--output", default="model_htp_compliant.onnx", help="Path to output model")
    
    args = parser.parse_args()
    
    enforce_htp_bool_constraints(args.model_path, args.spec_json_path, args.output)
