import os
import random
from typing import Any, Dict, List
import pandas as pd
import argparse
from tqdm import tqdm
from datetime import datetime
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import contextlib
import gc
import torch
import concurrent.futures
from enum import Enum       
from pydantic import BaseModel, constr 
import openai
import json
import ast
import vllm
class OfflineVLLMModel:
    def __init__(self, model_name: str, max_seq_len: int = 4096, dtype: str = "bfloat16", VLLM_TENSOR_PARALLEL_SIZE=1,
                 VLLM_GPU_MEMORY_UTILIZATION=0.95):
        random.seed(26)
        self.model_name = model_name
        self.llm = vllm.LLM(
            self.model_name,
            max_model_len=max_seq_len,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
            dtype=dtype,
            enforce_eager=False
        )
        self.tokenizer = self.llm.get_tokenizer()
                        
    def delete(self):
        del self.llm
        del self.tokenizer

    def batch_predict(self, msg: List[List[dict]]) -> List[str]:
        prompts_str = []
        error_indices = []

        for idx, conv in enumerate(msg):
            try:
                # Apply the chat template for formatting the messages properly
                formatted_prompt = self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
                prompts_str.append(formatted_prompt)
            except AttributeError as e:
                print(f"Error: {e}. Ensure that your tokenizer supports 'apply_chat_template'.")
                # Mark the index as an error and continue
                error_indices.append(idx)
                continue

        # Perform batch prediction only on successfully processed prompts
        predictions = self.batch_predict_str(prompts_str)

        # Insert "ERROR" in the predictions at the positions of the errors
        for idx in error_indices:
            predictions.insert(idx, "ERROR")

        return predictions

    def batch_predict_str(self,
                    prompts: List[str],
                    use_full_generation=True,
                    configs={}) -> List[str]:
        if use_full_generation:
            responses = self.llm.generate(
                prompts,
                vllm.SamplingParams(
                    n=configs.get('n', 1),
                    temperature=configs.get('temperature', 0.9),
                    top_p=configs.get('top_p', 0.7),  # top_p 参数,
                    seed=26,
                    skip_special_tokens=True,
                    max_tokens=configs.get('max_tokens', 1024),
                    stop=self.tokenizer.eos_token
                ),
                use_tqdm=False
            )
            batch_response = [self.process_response(response.outputs[0].text) for response in responses]
        else:
            responses = self.llm.generate(
                prompts,
                vllm.SamplingParams(
                    seed=26,
                    max_tokens=1,
                    logprobs=10     # can be 20
                ),
                use_tqdm=True
            )
            batch_response = [self.find_highest_prob_choice(response.outputs[0].logprobs[0], possible_choices) for response in responses]
        return batch_response

    def process_response(self, generation: str) -> str:
        generation = generation.strip()
        generation = self.extract_content(generation).strip()
        return generation

    @staticmethod
    def extract_content(s: str) -> str:
        start_tag = "<answer>"
        start_index = s.find(start_tag)
        if start_index == -1:
            return s
        else:
            return s[start_index + len(start_tag):]

    def find_highest_prob_choice(self, logprob_dict, possible_choices: List[str]) -> str:
        sorted_logprob = sorted(logprob_dict.values(), key=lambda x: x.logprob, reverse=True)
        sorted_tokens_logprobs = [(item.decoded_token, item.logprob) for item in sorted_logprob]
        chosen_option = None
        for token, logprob in sorted_tokens_logprobs:
            if len(token)>2:
                continue # 太长的token可排除
            for choice in possible_choices:
                if choice in token:
                    chosen_option = choice
                    break
            if chosen_option:
                break
        return chosen_option