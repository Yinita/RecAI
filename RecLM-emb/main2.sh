export NCCL_DEBUG=0 # 禁用 NCCL 的日志输出
export CUDA_VISIBLE_DEVICES=0,1,2,3
export QUERY_MAX_LEN=512
export TASK="xbox"

#----base
export INPUT_DIR="/home/aiscuser/figllm/toolcall/database/localdb/backup_data/emb_stage2/base"
export RESULT_DIR="output/xbox_base_stg2_infer/"
mkdir $RESULT_DIR

export OUT_DIR="$RESULT_DIR/test_bge_1019"
export MODEL_PATH_OR_NAME="/home/aiscuser/figllm/toolcall/database/localdb/backup_data/1019/models/reclm_emb_xbox_bge-m3_qwen_v1"
bash shell/infer_stage2.sh 
export OUT_DIR="$RESULT_DIR/test_base"
export MODEL_PATH_OR_NAME="BAAI/bge-m3"
bash shell/infer_stage2.sh 
export OUT_DIR="$RESULT_DIR/test_e5-large-v2_1007"
export MODEL_PATH_OR_NAME="/home/aiscuser/figllm/toolcall/database/localdb/emb_models/reclm_emb_xbox_e5-large-v2_1007"
bash shell/infer_stage2.sh 

# #----model rewrite

# export INPUT_DIR="/home/aiscuser/figllm/toolcall/database/localdb/backup_data/emb_stage2/specific-1029"
# export RESULT_DIR="output/xbox_rewrite_stg2_infer/"
# mkdir $RESULT_DIR

# export OUT_DIR="$RESULT_DIR/test_bge_1019"
# export MODEL_PATH_OR_NAME="/home/aiscuser/figllm/toolcall/database/localdb/backup_data/1019/models/reclm_emb_xbox_bge-m3_qwen_v1"
# bash shell/infer_stage2.sh 
# export OUT_DIR="$RESULT_DIR/test_base"
# export MODEL_PATH_OR_NAME="BAAI/bge-m3"
# bash shell/infer_stage2.sh 
# export OUT_DIR="$RESULT_DIR/test_e5-large-v2_1007"
# export MODEL_PATH_OR_NAME="/home/aiscuser/figllm/toolcall/database/localdb/emb_models/reclm_emb_xbox_e5-large-v2_1007"
# bash shell/infer_stage2.sh 


#----model rewrite v2

export INPUT_DIR="/home/aiscuser/figllm/toolcall/database/localdb/backup_data/emb_stage2/specific-1029-v2"
export RESULT_DIR="output/xbox_rewrite_stg2_v2_infer/"
mkdir $RESULT_DIR

export OUT_DIR="$RESULT_DIR/test_bge_1019"
export MODEL_PATH_OR_NAME="/home/aiscuser/figllm/toolcall/database/localdb/backup_data/1019/models/reclm_emb_xbox_bge-m3_qwen_v1"
bash shell/infer_stage2.sh 
export OUT_DIR="$RESULT_DIR/test_base"
export MODEL_PATH_OR_NAME="BAAI/bge-m3"
bash shell/infer_stage2.sh 
export OUT_DIR="$RESULT_DIR/test_e5-large-v2_1007"
export MODEL_PATH_OR_NAME="/home/aiscuser/figllm/toolcall/database/localdb/emb_models/reclm_emb_xbox_e5-large-v2_1007"
bash shell/infer_stage2.sh 


export OUT_DIR="infer_metric/test_bge_1019"
export MODEL_PATH_OR_NAME="/home/aiscuser/figllm/toolcall/database/localdb/backup_data/1019/models/reclm_emb_xbox_bge-m3_qwen_v1"
bash shell/infer_metrics.sh 