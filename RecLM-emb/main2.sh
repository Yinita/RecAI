#!/bin/bash
# Set API variables for vllm localhost setup
export NCCL_DEBUG=0 # 禁用 NCCL 的日志输出
export CUDA_VISIBLE_DEVICES=0,1,2,3

# export OPENAI_API_KEY="token-abc123"
# export OPENAI_API_BASE="http://localhost:8101/v1"
# export OPENAI_API_TYPE="onlinevllm"
# export OPENAI_API_VERSION="v1"
# export MODEL="Qwen/Qwen2.5-72B-Instruct"
# export batch_size=300
# export model_altname="qwen72B"
# export model_altname_v2="qwen72B_v2"

export OPENAI_API_BASE=https://gigaai.openai.azure.com/;
export OPENAI_API_VERSION="2024-05-01-preview"
export MODEL="gpt-4o"
export batch_size=300
export model_altname="gpt-4o"
export model_altname_v2="gpt-4o_v2"

export learning_rate=5e-5
export num_train_epochs=5
export QUERY_MAX_LEN=1024
export version="gpt_v1"
# model = gpt-4o
# model_altname = gpt4
export OUTPUT_DIR=output/xbox_gpt/reclm_emb_xbox_gpt_bge-m3_$version
export MODEL_NAME_OR_PATH="BAAI/bge-m3" # Currently support BAAI/bge-m3 (best)    intfloat/e5-large-v2, bert-large-uncased, BAAI/bge-large-en-v1.5, meta-llama/Llama-2-7b-hf
export RUN_NAME="bge_m3_$version"
export TASK="xbox_gpt"
bash shell/data_pipeline.sh
bash shell/test_data_pipeline.sh
bash shell/run_single_node.sh

export OUT_DIR="output/xbox_gpt_infer/$RUN_NAME"

cp -r /home/aiscuser/RecAI/RecLM-emb/data /home/aiscuser/figllm/toolcall/database/localdb/backup_data/1019

# export learning_rate=3e-5
# export num_train_epochs=5

# export QUERY_MAX_LEN=1024
# export version=2
# # model = Gpt-4-Turbo
# # model_altname = gpt4
# export OUTPUT_DIR=output/xbox_gpt/reclm_emb_xbox_gpt_bge-m3_$version
# export MODEL_NAME_OR_PATH="BAAI/bge-m3" # Currently support BAAI/bge-m3 (best)    intfloat/e5-large-v2, bert-large-uncased, BAAI/bge-large-en-v1.5, meta-llama/Llama-2-7b-hf
# export RUN_NAME="bge_m3_$version"
# export TASK="xbox_gpt"
# # bash shell/data_pipeline.sh
# # bash shell/test_data_pipeline.sh
# # bash shell/run_single_node.sh

# export OUT_DIR="output/xbox_gpt_infer/$RUN_NAME"
# export MODEL_PATH_OR_NAME=$OUTPUT_DIR
# # bash shell/infer_metrics.sh 
