export NCCL_DEBUG=0 # 禁用 NCCL 的日志输出
export task=user2item

python src/hn_mine.py \
    --model_name_or_path BAAI/bge-base-en-v1.5 \
    --input_file "/home/aiscuser/figllm/toolcall/database/localdb/recommendation_task/train/$task.jsonl" \
    --output_file "/home/aiscuser/${task}_minedHN.jsonl" \
    --range_for_sampling 2-200 \
    --negative_number 15 \
    --use_gpu_for_searching

    # --candidate_pool /home/aiscuser/yt/discoveryai_data/stage2_metric_data/metadata.jsonl \
