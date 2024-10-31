# #!/bin/bash
# # Set API variables for vllm localhost setup
export NCCL_DEBUG=0 # 禁用 NCCL 的日志输出
export CUDA_VISIBLE_DEVICES=0,1,2,3

export QUERY_MAX_LEN=512
export TASK="xbox"
# export OUT_DIR="output/xbox_infer/base"
# export MODEL_PATH_OR_NAME="BAAI/bge-m3"
# bash shell/infer_metrics.sh 

# export OUT_DIR="output/xbox_infer/ep1"
# export MODEL_PATH_OR_NAME=/home/aiscuser/RecAI/RecLM-emb/output/xbox/reclm_emb_xbox_bge-m3_qwen_v1/checkpoint-1774
# bash shell/infer_metrics.sh 

# export OUT_DIR="output/xbox_infer/test"
# export MODEL_PATH_OR_NAME=/home/aiscuser/figllm/toolcall/database/localdb/backup_data/1019/models/reclm_emb_xbox_bge-m3_qwen_v1
export OUT_DIR="output/xbox_infer/m3_1029_h30000-cp1800"
export MODEL_PATH_OR_NAME="/home/aiscuser/RecAI/RecLM-emb/output/xbox/bge-m3_1029_h30000/checkpoint-1792"
bash shell/infer_metrics.sh 

