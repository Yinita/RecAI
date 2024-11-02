export NCCL_DEBUG=0 # 禁用 NCCL 的日志输出
export CUDA_VISIBLE_DEVICES=0,1,2,3
export QUERY_MAX_LEN=512
export TASK="xbox"


export MODEL_PATH_OR_NAME="/home/aiscuser/yt/discoveryai_data/emb_model/bge-m3_v1101_1ep"


# #----base
# export INPUT_DIR="/home/aiscuser/RecAI/RecLM-emb/data/stage2/base/emb_stage2/base"
# export RESULT_DIR="output/xbox_base_stg2_infer/"
# mkdir $RESULT_DIR

# export OUT_DIR="$RESULT_DIR/m3_1101_1ep"
# bash shell/infer_stage2.sh 

#----model rewrite v3
# export INPUT_DIR="/home/aiscuser/RecAI/RecLM-emb/data/stage2/emb_stage2/specific-1031-v3"
# export RESULT_DIR="output/xbox_rewrite_stg2_v3_infer/"
# mkdir $RESULT_DIR

# export OUT_DIR="$RESULT_DIR/m3_1101_1ep"
# bash shell/infer_stage2.sh 

#----model rewrite v4
# export INPUT_DIR="/home/aiscuser/RecAI/RecLM-emb/data/stage2/emb_stage2/specific-1101-v4"
# export RESULT_DIR="output/xbox_rewrite_stg2_v4_infer/"
# mkdir $RESULT_DIR
# export OUT_DIR="$RESULT_DIR/m3_1101_1ep"
# bash shell/infer_stage2.sh 


export TASK="xbox"
export OUT_DIR="output/infer_metric/m3_1101_1ep"
bash shell/infer_metrics.sh 