import os
import json
from collections import defaultdict

# Define the base directory path
base_dir = '/home/aiscuser/RecAI/RecLM-emb/output'

# Define output data structure
results = defaultdict(lambda: defaultdict(dict))

# Function to process each all_metrics.jsonl file
def process_metrics_file(file_path, exp_name, model_name):
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                task_name = data.get("task_name")
                metrics = data.get("metrics", {})
                # Extract the score for recall if available in metrics
                score = {k: v for k, v in metrics.items() if "recall" in k}
                # Store the score in the results structure under appropriate task, model, and experiment
                if score:
                    results[task_name][model_name][exp_name] = score
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in {file_path}")

# Traverse the base directory
for root, dirs, files in os.walk(base_dir):
    for directory in dirs:
        if directory.endswith('infer'):
            exp_name = directory.split("_infer")[0].strip("_")
            exp_path = os.path.join(root, directory)
            
            # Traverse each experiment's model folder
            for model_dir in os.listdir(exp_path):
                model_path = os.path.join(exp_path, model_dir)
                if os.path.isdir(model_path):
                    model_name = model_dir
                    # Look for all_metrics.jsonl within each model folder
                    metrics_file = os.path.join(model_path, 'all_metrics.jsonl')
                    if os.path.isfile(metrics_file):
                        process_metrics_file(metrics_file, exp_name, model_name)

# Define the output path for the aggregated results
output_file_path = '/home/aiscuser/RecAI/RecLM-emb/analysis/models_compare.csv'

# Write the aggregated results to a JSON file
try:
    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)
except Exception as e:
    print(f"Error saving the results: {e}")

print("Metrics have been processed and saved.")
