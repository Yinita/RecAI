# pip install --upgrade accelerate
export NCCL_DEBUG=0 # 禁用 NCCL 的日志输出
# pip install --upgrade accelerate
# mkdir /home/aiscuser/FlagEmbedding/
# torchrun --nproc_per_node 4 \
# 	-m FlagEmbedding.finetune.embedder.encoder_only.m3 \
# 	--model_name_or_path BAAI/bge-m3 \
#     --cache_dir /home/aiscuser/cache/model \
#     --train_data /home/aiscuser/figllm/toolcall/database/localdb/recommendation_task/train \
#     --cache_path /home/aiscuser/cache/data \
#     --train_group_size 4 \
#     --query_max_len 512 \
#     --passage_max_len 512 \
#     --pad_to_multiple_of 8 \
#     --knowledge_distillation True \
#     --same_dataset_within_batch True \
#     --small_threshold 0 \
#     --drop_threshold 0 \
#     --output_dir /home/aiscuser/FlagEmbedding/1102_emb \
#     --overwrite_output_dir \
#     --learning_rate 1e-4 \
#     --fp16 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 30 \
#     --dataloader_drop_last True \
#     --warmup_ratio 0.1 \
#     --gradient_checkpointing \
#     --deepspeed /home/aiscuser/figllm/toolcall/database/localdb/recommendation_task/ds_stage0.json \
#     --logging_steps 1 \
#     --save_steps 1000 \
#     --negatives_cross_device \
#     --temperature 0.1 \
#     --sentence_pooling_method cls \
#     --normalize_embeddings True \
#     --kd_loss_type m3_kd_loss \
#     --unified_finetuning True \
#     --use_self_distill True \
#     --fix_encoder False \
#     --self_distill_start_step 0


# cd /home/aiscuser/RecAI/RecLM-emb
export QUERY_MAX_LEN=512
export TASK="xbox"
export OUT_DIR="output/infer_metric/test_1102-1400"

export MODEL_PATH_OR_NAME="/home/aiscuser/FlagEmbedding/1102_emb"
# bash shell/infer_metrics.sh 

# export INPUT_DIR="/home/aiscuser/figllm/toolcall/database/localdb/backup_data/emb_stage2/base"
export INPUT_DIR="/home/aiscuser/specific-1029-v2"

export RESULT_DIR="output/xbox_base_stg2_infer/"
export OUT_DIR="output/infer_metric/test_1102-1400"
bash shell/infer_stage2.sh 

# cp -r "output/infer_metric/test_1102-0240" /home/aiscuser/figllm/toolcall/database/localdb/recommendation_task/exp_result