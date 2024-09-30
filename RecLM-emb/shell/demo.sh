# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.



accelerate launch --config_file shell/infer_case.yaml demo.py \
    --in_meta_data data/xbox/metadata.json \
    --user_embedding_prompt_path output/demo/user_embedding_prompt.jsonl \
    --model_path_or_name "/home/aiscuser/remote_github/yinita/RecAI/RecLM-emb/output/xbox/reclm_emb_xbox_e5/checkpoint-418" \
    --topk 5 \
    --seed 2023 \
    --query_max_len 512 \
    --passage_max_len 280 \
    --per_device_eval_batch_size 1 \
    --sentence_pooling_method "mean" \
    --normlized \
    --has_template