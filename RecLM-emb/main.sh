#!/bin/bash
# Set API variables for vllm localhost setup
export NCCL_DEBUG=0 # 禁用 NCCL 的日志输出
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OPENAI_API_KEY="token-abc123"
export OPENAI_API_BASE="http://localhost:8101/v1"
export OPENAI_API_TYPE="onlinevllm"
export OPENAI_API_VERSION="v1"
export MODEL="Qwen/Qwen2.5-72B-Instruct"
export batch_size=300
export model_altname="qwen72B"
export model_altname_v2="qwen72B_v2"
export learning_rate=5e-5
export num_train_epochs=5
export QUERY_MAX_LEN=1024
export version=1
# model = Gpt-4-Turbo
# model_altname = gpt4
export OUTPUT_DIR=output/xbox/reclm_emb_xbox_bge-m3_$version
export MODEL_NAME_OR_PATH="BAAI/bge-m3" # Currently support BAAI/bge-m3 (best)    intfloat/e5-large-v2, bert-large-uncased, BAAI/bge-large-en-v1.5, meta-llama/Llama-2-7b-hf
export RUN_NAME="bge_m3_$version"
export TASK="xbox"
bash shell/data_pipeline.sh
bash shell/test_data_pipeline.sh
# bash shell/run_single_node.sh

export OUT_DIR="output/xbox_infer/$RUN_NAME"
export MODEL_PATH_OR_NAME=$OUTPUT_DIR
# bash shell/infer_metrics.sh 



export learning_rate=3e-5
export num_train_epochs=5

export QUERY_MAX_LEN=1024
export version=2
# model = Gpt-4-Turbo
# model_altname = gpt4
export OUTPUT_DIR=output/xbox/reclm_emb_xbox_bge-m3_$version
export MODEL_NAME_OR_PATH="BAAI/bge-m3" # Currently support BAAI/bge-m3 (best)    intfloat/e5-large-v2, bert-large-uncased, BAAI/bge-large-en-v1.5, meta-llama/Llama-2-7b-hf
export RUN_NAME="bge_m3_$version"
export TASK="xbox"
# bash shell/data_pipeline.sh
# bash shell/test_data_pipeline.sh
# bash shell/run_single_node.sh

export OUT_DIR="output/xbox_infer/$RUN_NAME"
export MODEL_PATH_OR_NAME=$OUTPUT_DIR
# bash shell/infer_metrics.sh 



# bash shell/data_pipeline.sh
# bash shell/test_data_pipeline.sh
# export QUERY_MAX_LEN=512
# export OUTPUT_DIR=output/xbox/e5-v1
# export MODEL_NAME_OR_PATH="intfloat/e5-large-v2" # Currently support BAAI/bge-m3 (best)    intfloat/e5-large-v2, bert-large-uncased, BAAI/bge-large-en-v1.5, meta-llama/Llama-2-7b-hf
# export RUN_NAME="e5-large-v2_$version"
# export TASK="xbox"
# bash shell/run_single_node.sh

# export OUT_DIR="output/xbox_infer/e5-$version"
# export MODEL_PATH_OR_NAME=output/xbox/e5-$version
# bash shell/infer_metrics.sh 

# 遍历指定目录下的所有子目录
# for sub_folder in /home/aiscuser/RecAI/RecLM-emb/output/xbox/*; do
#     if [ -d "$sub_folder" ]; then
#         sub_folder_name=$(basename "$sub_folder")
#         export OUT_DIR="output/xbox_infer/$sub_folder_name"
#         export MODEL_PATH_OR_NAME="$sub_folder"
#         bash shell/infer_metrics.sh
#     fi
# done

# export version=v5_50_200k
# # model = Gpt-4-Turbo
# # model_altname = gpt4
# export OUTPUT_DIR=output/xbox/reclm_emb_xbox_bge-m3_$version
# export MODEL_NAME_OR_PATH="BAAI/bge-m3" # Currently support BAAI/bge-m3 (best)    intfloat/e5-large-v2, bert-large-uncased, BAAI/bge-large-en-v1.5, meta-llama/Llama-2-7b-hf
# export RUN_NAME="reclm_emb_xbox_bge-m3_$version"
# export TASK="xbox"
# # bash shell/data_pipeline.sh
# # bash shell/test_data_pipeline.sh
# bash shell/run_single_node.sh

# export OUT_DIR="output/xbox_infer/bge-m3_$version"
# export MODEL_PATH_OR_NAME=output/xbox/reclm_emb_xbox_bge-m3_$version
# bash shell/infer_metrics.sh 