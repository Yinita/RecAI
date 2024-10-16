# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#### example for training LM on xbox data

# export WANDB_DISABLED=true;
export DISABLE_MLFLOW_INTEGRATION=true;
export WANDB_DIR=$HOME/.cache/
export WANDB_PROJECT="RecLM-emb"


DATA_DIR=data/xbox/train

# MODEL_NAME_OR_PATH="BAAI/bge-m3" # Currently support BAAI/bge-m3 (best)    intfloat/e5-large-v2, bert-large-uncased, BAAI/bge-large-en-v1.5, meta-llama/Llama-2-7b-hf
SENTENCE_POOLING_METHOD="mean" # mean: intfloat/e5-large-v2 and bert-large-uncased; cls: BAAI/bge-large-en-v1.5 ; last: meta-llama/Llama-2-7b-hf
PASSAGE_MAX_LEN=128
GRADIENT_ACCU_STEPS=8

if [ "$MODEL_NAME_OR_PATH" = "meta-llama/Llama-2-7b-hf" ]; then
    TORCH_TYPE="bfloat16"
    BF16=True
    ATTN_IMPLEMENTATION="flash_attention_2"
    TRAIN_GROUP_SIZE=2
    PEFT_MODEL_NAME="castorini/repllama-v1-7b-lora-passage"
    HAS_TEMPLATE=True

    # ,$DATA_DIR/vaguequery2item.jsonl
    torchrun --nnodes=1 --nproc_per_node 4 --master_port=29502 train.py \
        --torch_dtype $TORCH_TYPE \
        --bf16 $BF16 \
        --attn_implementation $ATTN_IMPLEMENTATION \
        --train_group_size $TRAIN_GROUP_SIZE \
        --peft_model_name $PEFT_MODEL_NAME \
        --has_template $HAS_TEMPLATE \
        --data_cache_dir $HOME/.cache/hf_data \
        --output_dir $OUTPUT_DIR \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --train_data $DATA_DIR/gpt_data_v2.jsonl,$DATA_DIR/misspell2item.jsonl,$DATA_DIR/negquery2item.jsonl,$DATA_DIR/relativequery2item.jsonl,$DATA_DIR/title2item.jsonl,$DATA_DIR/gpt_data.jsonl,$DATA_DIR/item2item.jsonl,$DATA_DIR/query2item.jsonl,$DATA_DIR/queryuser2item.jsonl,$DATA_DIR/user2item.jsonl \
        --learning_rate 3e-5 \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --dataloader_drop_last True \
        --normlized True \
        --sentence_pooling_method $SENTENCE_POOLING_METHOD \
        --temperature 0.01 \
        --query_max_len $QUERY_MAX_LEN \
        --passage_max_len $PASSAGE_MAX_LEN \
        --dataloader_num_workers=2 \
        --gradient_accumulation_steps $GRADIENT_ACCU_STEPS \
        --logging_steps 100 \
        --save_strategy epoch \
        --warmup_ratio 0.1 \
        --report_to wandb \
        --run_name $RUN_NAME > ./training.log 2>&1

else
    TRAIN_GROUP_SIZE=4
    HAS_TEMPLATE=True

    torchrun --nnodes=1 --nproc_per_node 4 --master_port=29502 train.py \
        --train_group_size $TRAIN_GROUP_SIZE \
        --has_template $HAS_TEMPLATE \
        --data_cache_dir $HOME/.cache/hf_data \
        --output_dir $OUTPUT_DIR \
        --model_name_or_path $MODEL_NAME_OR_PATH \
        --train_data $DATA_DIR/gpt_data_v2.jsonl,$DATA_DIR/misspell2item.jsonl,$DATA_DIR/negquery2item.jsonl,$DATA_DIR/relativequery2item.jsonl,$DATA_DIR/title2item.jsonl,$DATA_DIR/gpt_data.jsonl,$DATA_DIR/item2item.jsonl,$DATA_DIR/query2item.jsonl,$DATA_DIR/queryuser2item.jsonl,$DATA_DIR/user2item.jsonl \
        --learning_rate $learning_rate \
        --num_train_epochs $num_train_epochs \
        --per_device_train_batch_size 4 \
        --dataloader_drop_last True \
        --normlized True \
        --sentence_pooling_method $SENTENCE_POOLING_METHOD \
        --temperature 0.01 \
        --query_max_len $QUERY_MAX_LEN \
        --passage_max_len $PASSAGE_MAX_LEN \
        --dataloader_num_workers=2 \
        --gradient_accumulation_steps $GRADIENT_ACCU_STEPS \
        --logging_steps 100 \
        --save_strategy epoch \
        --warmup_ratio 0.1 \
        --report_to wandb \
        --run_name $RUN_NAME > ./training.log 2>&1

fi

