import json
import os
import sys
import argparse
import torch

# Add current directory to sys.path to ensure we can import cr_evaluator
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the evaluator class
try:
    from cr_evaluator import CompressionEvaluator
except ImportError:
    # If running from a different directory, try to append the path
    # Assuming the script is located at Infer_Comm/evaluation/param_search/
    sys.path.append(os.path.join(os.getcwd(), 'Infer_Comm', 'evaluation', 'param_search'))
    from cr_evaluator import CompressionEvaluator

def main():
    parser = argparse.ArgumentParser(description="Calculate and update Compression Ratio (CR) in a JSON config file.")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the JSON file containing configurations.")
    parser.add_argument("--model_name", type=str, default="Meta-Llama-3.1-8B-Instruct", help="Name of the model to use.")
    parser.add_argument("--task", type=str, default="2wikimqa", help="Task name for data loading (e.g., qasper, 2wikimqa).")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for calculation.")
    
    args = parser.parse_args()
    
    json_file_path = args.json_file
    
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at {json_file_path}")
        return

    print(f"Reading configurations from {json_file_path}...")
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    if not isinstance(configs, list):
        print("Error: JSON content is not a list of configurations.")
        return

    print(f"Initializing CompressionEvaluator with model='{args.model_name}' and task='{args.task}'...")
    try:
        evaluator = CompressionEvaluator(model_name=args.model_name, task=args.task, device=args.device)
    except Exception as e:
        print(f"Failed to initialize evaluator: {e}")
        # Print sys.path to help debug import issues if any
        print("Current sys.path:", sys.path)
        return

    print(f"Processing {len(configs)} configurations...")
    
    modified_count = 0
    for i, config in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] Processing config_id: {config.get('config_id', i)}...")
        
        # Prepare params for the evaluator
        # Ensure keys match what cr_evaluator expects
        try:
            params = {
                "transform_type": config.get("transform_type", "hadamard"),
                "heads_selection": config["heads_selection"],
                "high_key_max_value": config["high_key_max_value"],
                "high_value_max_value": config["high_value_max_value"],
                "low_key_max_value": config["low_key_max_value"],
                "low_value_max_value": config["low_value_max_value"],
                "axis_key": config["axis_key"],
                "axis_value": config["axis_value"],
            }
            
            # Calculate CR
            # Note: evaluate returns a float
            new_cr = evaluator.evaluate(params)
            
            # Update the configuration
            old_cr = config.get("cr", "N/A")
            print(f"    -> Old CR: {old_cr} | New CR: {new_cr}")
            config["cr"] = new_cr
            modified_count += 1
            
        except KeyError as e:
            print(f"    -> Skipping config due to missing key: {e}")
        except Exception as e:
            print(f"    -> Error evaluating config: {e}")

    if modified_count > 0:
        # Overwrite the file with updated data
        print(f"Saving updated configurations back to {json_file_path}...")
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(configs, f, indent=4, ensure_ascii=False)
        print("Completed successfully.")
    else:
        print("No configurations were modified.")

if __name__ == "__main__":
    main()

