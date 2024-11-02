# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# step1:  1.metric  2.data 组成  3. query-block目的 rec: history->items/blocks; search: misspell,...->items/blocks

RAW_DATA_DIR="data/xbox/"

ALL_METRICS_FILE=$OUT_DIR/all_metrics.jsonl
TOPK="[5, 10]"
SEED=2024
QUERY_MAX_LEN=1024
PASSAGE_MAX_LEN=512
SENTENCE_POOLING_METHOD="mean"

if [ "$MODEL_PATH_OR_NAME" = "ada_embeddings" ] || [ "$MODEL_PATH_OR_NAME" = "text-embedding-ada-002" ] || [ "$MODEL_PATH_OR_NAME" = "text-embedding-3-large" ]; then
    echo "using openai model"
    CONFIG_FILE=./shell/infer_case.yaml
else
    echo "using huggingface model"
    CONFIG_FILE=./shell/infer.yaml
fi
# $INPUT_DIR/stage2_eval_8B_specific_v1_1028.jsonl  $TEST_DATA_DIR/user2item.jsonl \
echo "infer Recall"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $INPUT_DIR/Recall.jsonl \
    --answer_file $OUT_DIR/Recall.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "[5]" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 1024 \
    --task_type "stage2" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized
echo "infer Rank"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $INPUT_DIR/Rank.jsonl \
    --answer_file $OUT_DIR/Rank.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "[10]" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 1024 \
    --task_type "stage2" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized
echo "infer QA_Title_Attribute_key2val"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $INPUT_DIR/QA_Title_Attribute_key2val.jsonl \
    --answer_file $OUT_DIR/QA_Title_Attribute_key2val.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "[10]" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 1024 \
    --task_type "stage2" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized
echo "infer QA_Attributes2Title"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $INPUT_DIR/QA_Attributes2Title.jsonl \
    --answer_file $OUT_DIR/QA_Attributes2Title.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "[10]" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 1024 \
    --task_type "stage2" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized
echo "infer Co-play_Title2Items"
accelerate launch --config_file $CONFIG_FILE infer_metrics.py \
    --in_seq_data $RAW_DATA_DIR/sequential_data.txt \
    --in_meta_data $RAW_DATA_DIR/metadata.json \
    --model_path_or_name $MODEL_PATH_OR_NAME \
    --user_embedding_prompt_path $INPUT_DIR/Co-play_Title2Items.jsonl \
    --answer_file $OUT_DIR/Co-play_Title2Items.jsonl \
    --all_metrics_file $ALL_METRICS_FILE \
    --topk "[10]" \
    --seed $SEED \
    --query_max_len $QUERY_MAX_LEN \
    --passage_max_len $PASSAGE_MAX_LEN \
    --per_device_eval_batch_size 1024 \
    --task_type "stage2" \
    --sentence_pooling_method $SENTENCE_POOLING_METHOD \
    --normlized