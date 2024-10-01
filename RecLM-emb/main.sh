#!/bin/bash
# Set API variables for vllm localhost setup
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OPENAI_API_KEY="token-abc123"
export OPENAI_API_BASE="http://localhost:8101/v1"
export OPENAI_API_TYPE="onlinevllm"
export OPENAI_API_VERSION="v1"
export MODEL="Qwen/Qwen2.5-72B-Instruct"
export model_altname="qwen72B"
# model = Gpt-4-Turbo
# model_altname = gpt4
export OUTPUT_DIR=output/xbox/reclm_emb_xbox_bge-m3_v2
export MODEL_NAME_OR_PATH="BAAI/bge-m3" # Currently support BAAI/bge-m3 (best)    intfloat/e5-large-v2, bert-large-uncased, BAAI/bge-large-en-v1.5, meta-llama/Llama-2-7b-hf
export RUN_NAME="reclm_emb_xbox_bge-m3_v2"
export TASK="xbox"
bash shell/data_pipeline.sh
bash shell/test_data_pipeline.sh
bash shell/run_single_node.sh

export OUT_DIR="output/xbox_infer/bge-m3-base_v2"
export MODEL_PATH_OR_NAME="output/xbox/reclm_emb_xbox_bge-m3_v2"
bash shell/infer_metrics.sh 
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