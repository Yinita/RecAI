#!/bin/bash
# Set API variables for vllm localhost setup
export NCCL_DEBUG=0 # 禁用 NCCL 的日志输出
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OPENAI_API_KEY="token-abc123"
export OPENAI_API_BASE="http://localhost:8101/v1"
export OPENAI_API_TYPE="onlinevllm"
export OPENAI_API_VERSION="v1"
export MODEL="Qwen/Qwen2.5-72B-Instruct"
export batch_size=250
export model_altname="qwen72B"
export model_altname_v2="qwen72B_v2"
export learning_rate=5e-5
export num_train_epochs=4

export QUERY_MAX_LEN=1024
# model = Gpt-4-Turbo
# model_altname = gpt4
export OUTPUT_DIR=output/xbox/reclm_emb_xbox_bge-m3_v5
export MODEL_NAME_OR_PATH="BAAI/bge-m3" # Currently support BAAI/bge-m3 (best)    intfloat/e5-large-v2, bert-large-uncased, BAAI/bge-large-en-v1.5, meta-llama/Llama-2-7b-hf
export RUN_NAME="reclm_emb_xbox_bge-m3_v5"
export TASK="xbox"
bash shell/data_pipeline.sh
bash shell/test_data_pipeline.sh
# bash shell/run_single_node.sh

# # 遍历指定目录下的所有子目录
# for sub_folder in /home/aiscuser/RecAI/RecLM-emb/output/xbox/*; do
#     if [ -d "$sub_folder" ]; then
#         sub_folder_name=$(basename "$sub_folder")
#         export OUT_DIR="output/xbox_infer/$sub_folder_name"
#         export MODEL_PATH_OR_NAME="$sub_folder"
#         bash shell/infer_metrics.sh
#     fi
# done

# # 遍历指定的模型列表
# for model in "BAAI/bge-m3" "intfloat/e5-large-v2"; do
#     export OUT_DIR="output/xbox_infer/$model"
#     export MODEL_PATH_OR_NAME="$model"
#     bash shell/infer_metrics.sh
# done

#!/bin/bash

# for sub_folder in $(find /home/aiscuser/RecAI/RecLM-emb/output/xbox/reclm_emb_xbox_bge-m3_v5 -mindepth 1 -maxdepth 1 -type d)
# do
#     sub_folder_name=$(basename $sub_folder)
#     export OUT_DIR="output/xbox_infer/bge-m3_v5_$sub_folder_name"
#     export MODEL_PATH_OR_NAME="$sub_folder"
#     bash shell/infer_metrics.sh
# done

# export OUT_DIR="output/xbox_infer/bge-m3-base"
# export MODEL_PATH_OR_NAME="BAAI/bge-m3"
# bash shell/infer_metrics.sh 

# export OUT_DIR="output/xbox_infer/e5-large-v2-base"
# export MODEL_PATH_OR_NAME="intfloat/e5-large-v2"
# bash shell/infer_metrics.sh 
# export OUT_DIR="output/xbox_infer/e5-large-v2"
# export MODEL_PATH_OR_NAME="xboxoutput/xbox/reclm_emb_xbox_e5"
# bash shell/infer_metrics.sh 


# export OUT_DIR="output/xbox_infer/bert-large-uncased-base"
# export MODEL_PATH_OR_NAME="bert-large-uncased"
# bash shell/infer_metrics.sh 
# export OUT_DIR="output/xbox_infer/bert-large-uncased"
# export MODEL_PATH_OR_NAME="xboxoutput/xbox/reclm_emb_xbox_bert"
# bash shell/infer_metrics.sh 