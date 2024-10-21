import os
import json

def check_and_remove_neg_lines(folder_path):
    # 遍历文件夹中的所有 .jsonl 文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jsonl"):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                modified_lines = []
                has_modifications = False

                # 逐行读取文件内容
                with open(file_path, 'r') as f:
                    for i, line in enumerate(f):
                        try:
                            data = json.loads(line.strip())
                            # 检查是否有 'neg' 字段且为 None 或者空列表
                            if 'neg' in data and (data['neg'] is None or data['neg'] == []):
                                print(f"Removing line {i+1} in file: {file_path}")
                                has_modifications = True
                                continue  # 跳过当前行，即删除这一行
                            modified_lines.append(json.dumps(data))
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON in file: {file_path}, line: {i+1}")

                # 如果有修改，覆盖原文件
                if has_modifications:
                    with open(file_path, 'w') as f:
                        f.write("\n".join(modified_lines) + "\n")
                    print(f"File updated: {file_path}")
                else:
                    print(f"No modifications needed for file: {file_path}")

def main(data_root):
    train_folder = os.path.join(data_root, 'train')
    eval_folder = os.path.join(data_root, 'eval')

    # 检查并处理 train 文件夹
    print("Processing train folder...")
    check_and_remove_neg_lines(train_folder)

    # 检查并处理 eval 文件夹
    print("Processing eval folder...")
    check_and_remove_neg_lines(eval_folder)

if __name__ == "__main__":
    # 传入 data root 路径
    data_root = '/home/aiscuser/RecAI/RecLM-emb/data/xbox'
    main(data_root)
