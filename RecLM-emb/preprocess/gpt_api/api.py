# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import os
import time
import argparse
import pandas as pd
import os.path as osp
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI, AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider, AzureCliCredential
# Define a simple cache dictionary
model_cache = {}

def load_vllm_model(api_type: str, model_name: str, VLLM_TENSOR_PARALLEL_SIZE=1):
    if api_type == "onlinevllm":
        from vllm_server import OfflineVLLMModel
        if model_name in model_cache:
            print("Loading model from cache...")
            return model_cache[model_name]
        else:
            print("Loading model from disk...")
            model = OfflineVLLMModel(model_name=model_name, VLLM_TENSOR_PARALLEL_SIZE=VLLM_TENSOR_PARALLEL_SIZE)
            # Cache the loaded model
            model_cache[model_name] = model
            return model
    else:
        raise ValueError("Invalid API type")
    
api_key = os.environ.get('OPENAI_API_KEY') if os.environ.get('OPENAI_API_KEY') else None
api_base =  os.environ.get('OPENAI_API_BASE') if os.environ.get('OPENAI_API_BASE') else None
api_type = os.environ.get('OPENAI_API_TYPE') if os.environ.get('OPENAI_API_TYPE') else None
api_version =  os.environ.get('OPENAI_API_VERSION') if os.environ.get('OPENAI_API_VERSION') else None

if api_key:
    if api_type == "azure":
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_base
        )
    elif api_type == "offlinevllm":
        client = OpenAI(api_key=api_key,
                                        base_url=api_base)
    else:
        client = OpenAI(  
            api_key=api_key
        )
else:
    credential = AzureCliCredential()    

    token_provider = get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default"
    )

    client = AzureOpenAI(
        azure_endpoint=api_base,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
        max_retries=5,
    )

MODEL = os.environ.get('MODEL')


def call_chatgpt(prompt):
    max_retry_cnt = 5
    result = "NULL"
    tokens = []
    for i in range(max_retry_cnt):
        try:
            response = client.chat.completions.create(
                model=MODEL,  
                messages=[
                    {"role": "system",
                    "content": "You are a helpful assistant. \n"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                temperature=1.0,
                top_p=1.0,
            )
            result = response.choices[0].message.content
            tokens = [response.usage.prompt_tokens, response.usage.completion_tokens]

            break
        except Exception as e:
            error_msg = str(e)
            print(f"OpenAI API Error: {error_msg}")            
            if "content filtering" in error_msg:
                break            
            if "time" in error_msg or "exceeded token rate limit" in error_msg:
                print("Rate limit reached. Waiting for 20 seconds...")
                time.sleep(20) 
    if not result:
        result = "NULL"
        tokens = [0,0]
    return result, tokens

def process_row(writer, sample):
    question = sample['question'] 
    output, tokens = call_chatgpt(question)
    writer.writerow([question, output])
    # Assuming `call_chatgpt` returns input/output token numbers
    return tokens[0], tokens[1]



def process_hf_data(dataset, output_file, args):
    filename = os.path.basename(output_file)
    total_input_token_num, total_output_token_num = 0, 0

    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['question', 'response'])
        
        try:
            if api_type == "onlinevllm":
                model = load_vllm_model(api_type, MODEL, VLLM_TENSOR_PARALLEL_SIZE=2)
                his = []
                batch_size = 250
                dataset_length = len(dataset)  # Get the total length of the dataset

                for i, sample in tqdm(enumerate(dataset), desc=filename):
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant. \n"},
                        {"role": "user", "content": sample['question']},
                    ]
                    his.append(messages)
                    if (i + 1) % batch_size == 0 or i + 1 == dataset_length:
                        output = model.batch_predict(his)

                        for idx, o in enumerate(output):
                            data_index = i + 1 - len(his) + idx  # Ensure we don't go out of bounds
                            if data_index < dataset_length:  # Check to avoid IndexError
                                writer.writerow([dataset[data_index]["question"], o])

                        his = []  # Reset history after batch processing
            else:
                for i, sample in tqdm(enumerate(dataset), desc=filename):
                    
                    input_token_num, output_token_num = process_row(writer, sample)
                    total_input_token_num += input_token_num
                    total_output_token_num += output_token_num
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, exiting...")
            print(f"Total tokens processed so far: input - {total_input_token_num}, output - {total_output_token_num}")

    return total_input_token_num, total_output_token_num

## return a list of dictionaries
def load_jsonl_from_disk(file_path):
    ## read a csv file with pd, the first line is header 
    df = pd.read_csv(file_path)    
    return df.to_dict(orient='records')


def main(args):
    total_token_num, total_cost = 0, 0

    infile = args.input_file
    outfile = args.output_file
    data_as_list = load_jsonl_from_disk(infile)

    total_input_token_num, total_output_token_num = process_hf_data(data_as_list, outfile, args)
    # cost = 0.015 * total_input_token_num / 1000 + 0.0020 * total_output_token_num / 1000
    # cost = 0.01 * total_input_token_num / 1000 + 0.03 * total_output_token_num / 1000
    cost = 10 * total_input_token_num / 1000000 + 30 * total_output_token_num / 1000000
    total_token_num = total_input_token_num + total_output_token_num
    print(">> Task done. Use {:d} tokens in total, and cost $ {:.4f}.".format(total_token_num, cost)) 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_process", type=int, default=1)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()
    main(args)
