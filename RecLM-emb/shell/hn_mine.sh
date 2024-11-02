export NCCL_DEBUG=0 # 禁用 NCCL 的日志输出

export task=user2item
python src/hn_mine.py \
    --model_name_or_path BAAI/bge-m3 \
    --input_file "/home/aiscuser/figllm/toolcall/database/localdb/recommendation_task/train/$task.jsonl" \
    --output_file "/home/aiscuser/${task}_minedHN.jsonl" \
    --range_for_sampling 10-210 \
    --negative_number 15 \
    --use_gpu_for_searching

# python src/add_reranker_score.py \
#     --input_file "/home/aiscuser/figllm/toolcall/database/localdb/recommendation_task/train/$task.jsonl" \
#     --output_file "/home/aiscuser/${task}_finetune_data_score.jsonl" \
#     --range_for_sampling 10-210 \
#     --negative_number 15 \
#     --use_gpu_for_searching 

export task=item2item
python src/hn_mine.py \
    --model_name_or_path BAAI/bge-m3 \
    --input_file "/home/aiscuser/figllm/toolcall/database/localdb/recommendation_task/train/$task.jsonl" \
    --output_file "/home/aiscuser/${task}_minedHN.jsonl" \
    --range_for_sampling 10-210 \
    --negative_number 15 \
    --use_gpu_for_searching

    # --candidate_pool /home/aiscuser/yt/discoveryai_data/stage2_metric_data/metadata.jsonl \
