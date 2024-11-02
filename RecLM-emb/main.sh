#!/bin/bash
# Set API variables for vllm localhost setup
export NCCL_DEBUG=0 # 禁用 NCCL 的日志输出
export CUDA_VISIBLE_DEVICES=0,1,2,3

export OPENAI_API_KEY="token-abc123"
export OPENAI_API_BASE="http://localhost:8101/v1"
export OPENAI_API_TYPE="onlinevllm"
export OPENAI_API_VERSION="v1"
export MODEL="Qwen/Qwen2.5-72B-Instruct"
export batch_size=1024
export model_altname="qwen72B"
export model_altname_v2="qwen72B_v2"

# export OPENAI_API_BASE=https://gigaai.openai.azure.com/;
# export OPENAI_API_VERSION="2024-05-01-preview"
# export MODEL="gpt-4o"
# export batch_size=300
# export model_altname="gpt-4o"
# export model_altname_v2="gpt-4o_v2"

export learning_rate=1e-4
export num_train_epochs=3
export QUERY_MAX_LEN=1024
export version="v1102"
# model = gpt-4o
# model_altname = gpt4
export OUTPUT_DIR=output/xbox/bge-m3_$version
export MODEL_NAME_OR_PATH="BAAI/bge-m3" # Currently support BAAI/bge-m3 (best)    intfloat/e5-large-v2, bert-large-uncased, BAAI/bge-large-en-v1.5, meta-llama/Llama-2-7b-hf
export RUN_NAME="bge_m3_$version"
export TASK="xbox"

# mkdir /home/aiscuser/RecAI/RecLM-emb/data
# cp -r /home/aiscuser/figllm/toolcall/database/localdb/backup_data/1019/xbox /home/aiscuser/RecAI/RecLM-emb/data


bash shell/data_pipeline.sh
bash shell/test_data_pipeline.sh
python /home/aiscuser/RecAI/RecLM-emb/preprocess/data_clean.py
export BATCH_SIZE=4
# # cp -r /home/aiscuser/RecAI/RecLM-emb/output /home/aiscuser/figllm/toolcall/database/localdb/backup_data/1029_h30000
bash shell/run_single_node.sh
export OUT_DIR="output/xbox_infer/$RUN_NAME"
export MODEL_PATH_OR_NAME=$OUTPUT_DIR
bash shell/infer_metrics.sh 
# # mkdir /home/aiscuser/figllm/toolcall/database/localdb/backup_data/1029_h30000
# # cp -r /home/aiscuser/RecAI/RecLM-emb/output /home/aiscuser/figllm/toolcall/database/localdb/backup_data/1029_h30000

# # 基础路径
# OUTPUT_BASE_PATH="/home/aiscuser/RecAI/RecLM-emb/output/xbox"
# INFER_BASE_PATH="output/xbox_infer"

# # 遍历指定目录下的所有子目录
# for sub_folder in "$OUTPUT_BASE_PATH"/*; do
#     if [ -d "$sub_folder" ]; then
#         for checkpoint_folder in "$sub_folder"/checkpoint*; do
#             if [ -d "$checkpoint_folder" ]; then
#                 # 提取 checkpoint 文件夹的名称
#                 checkpoint_name=$(basename "$checkpoint_folder")
                
#                 # 设置推理输出目录为 xbox_infer/checkpoint_name
#                 export OUT_DIR="${INFER_BASE_PATH}/${checkpoint_name}"
#                 export MODEL_PATH_OR_NAME="$checkpoint_folder"
                
#                 # 运行推理脚本
#                 bash shell/infer_metrics.sh
#             fi
#         done
#     fi
# done

export learning_rate=1e-4
export num_train_epochs=3
export QUERY_MAX_LEN=1024
export version="v1102_v2"
# model = gpt-4o
# model_altname = gpt4
export OUTPUT_DIR=output/xbox/bge-m3_$version
export MODEL_NAME_OR_PATH="BAAI/bge-m3" # Currently support BAAI/bge-m3 (best)    intfloat/e5-large-v2, bert-large-uncased, BAAI/bge-large-en-v1.5, meta-llama/Llama-2-7b-hf
export RUN_NAME="bge_m3_$version"
export TASK="xbox"
export BATCH_SIZE=6
bash shell/run_single_node.sh
export OUT_DIR="output/xbox_infer/$RUN_NAME"
export MODEL_PATH_OR_NAME=$OUTPUT_DIR
bash shell/infer_metrics.sh 