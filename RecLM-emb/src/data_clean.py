import os
import json
import sys

import os
import json
import sys

def clean_jsonl_file(file_path):
    cleaned_data = []
    changes_made = []

    required_fields = {"query", "pos", "neg", "pos_scores", "neg_scores", "prompt", "type"}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                changes_made.append(f"Skipped malformed line in {file_path}.")
                continue

            query = data.get("query", "")
            pos = data.get("pos", [])
            neg = data.get("neg", [])
            pos_scores = data.get("pos_scores", [])
            neg_scores = data.get("neg_scores", [])
            prompt = data.get("prompt", "")
            dtype = data.get("type", "")

            modified = False

            # Check and correct 'query'
            if not query or not pos or not neg:
                changes_made.append(f"Removed line with empty query/pos/neg in {file_path}.")
                continue  # Skip lines with empty or missing 'query'

            if not isinstance(query, str):
                changes_made.append(f"Corrected 'query' type in {file_path}.")
                query = str(query)
                modified = True

            # Check and correct 'pos'
            if not isinstance(pos, list):
                changes_made.append(f"Corrected 'pos' type in {file_path}.")
                pos = [pos] if isinstance(pos, str) else []
                modified = True

            # Check and correct 'neg'
            if not isinstance(neg, list):
                changes_made.append(f"Corrected 'neg' type in {file_path}.")
                neg = [neg] if isinstance(neg, str) else []
                modified = True

            # Filter only required fields
            # cleaned_entry = {k: v for k, v in data.items() if k in required_fields}
            cleaned_entry = {}
            # Update the data dictionary with corrected values
            cleaned_entry.update({
                "query": query,
                "pos": pos,
                "neg": neg,
                # "pos_scores": pos_scores,
                # "neg_scores": neg_scores,
                # "prompt": prompt,
                # "type": dtype
            })

            cleaned_data.append(cleaned_entry)

    # Write cleaned data back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        for data in cleaned_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    # Print changes made
    for change in changes_made:
        print(change)

def clean_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")
            clean_jsonl_file(file_path)

if __name__ == "__main__":
    # Check if user has provided a path
    if len(sys.argv) < 2:
        print("Usage: python data_clean.py <file_or_folder_path>")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isdir(path):
        clean_folder(path)
    elif os.path.isfile(path) and path.endswith(".jsonl"):
        clean_jsonl_file(path)
    else:
        print("Provided path is neither a JSONL file nor a folder containing JSONL files.")
