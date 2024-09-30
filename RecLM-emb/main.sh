#!/bin/bash
# Set API variables for vllm localhost setup

export OPENAI_API_KEY="token-abc123"
export OPENAI_API_BASE="http://localhost:8101/v1"
export OPENAI_API_TYPE="onlinevllm"
export OPENAI_API_VERSION="v1"
export MODEL="Qwen/Qwen2.5-72B-Instruct"


export TASK="xbox"
bash shell/data_pipeline.sh
bash shell/test_data_pipeline.sh