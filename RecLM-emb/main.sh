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

export TASK="xbox"
# bash shell/data_pipeline.sh
# bash shell/test_data_pipeline.sh
bash shell/run_single_node.sh