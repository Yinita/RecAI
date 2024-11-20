export ALLOW_REGENERATE="true"
# pip install -r requirements.txt
# set up env

export NCCL_DEBUG=0 # 禁用 NCCL 的日志输出
python ./preprocess/generate_data.py --tasks retrieval,ranking,explanation --sample_num 10000 --dataset steam
# set up data


# task:  ranking, retrieval, explanation, conversation, embedding_ranking, embedding_retrieval, chatbot  
# tasks=("ranking" "retrieval")

# for task in "${tasks[@]}"
#     do
#         echo "Running task: $task"
#         python eval.py --task-names $task \
#             --bench-name steam \
#             --model_path_or_name Qwen/Qwen2.5-7B-Instruct \
#             --batch_size 10000
#     done
