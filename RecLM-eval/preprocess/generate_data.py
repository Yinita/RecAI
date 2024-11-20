# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import re
import json
import gzip
import math
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from multiprocessing import Pool
from collections import defaultdict
random.seed(42)

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def ReadLineFromFile(path):
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def parse(path):
    if path.endswith('.gz'):
        g = gzip.open(path, 'rt') 
    else:
        g = open(path, 'r')
    
    # 逐行解析
    for l in g:
        try:
            yield json.loads(l.strip()) 
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON line: {l.strip()} - {e}")
            continue 

def negative_samples_maker(sequential_data, sample_num=100000):
    print("Generate negative samples")
    all_games = set()
    user_sequences = []

    for sequence in sequential_data:
        items = sequence.split()
        user_id, games = items[0], items[1:]
        user_sequences.append((user_id, set(games)))
        all_games.update(games) 

    all_games = list(all_games)
    negative_samples = []

    for user_id, user_games in tqdm(user_sequences[:sample_num]):
        candidate_negatives = list(set(all_games) - user_games)
        if len(candidate_negatives) >= 20:
            sampled_negatives = random.sample(candidate_negatives, 20)
        else:
            sampled_negatives = candidate_negatives 
        negative_samples.append(f"{user_id} {' '.join(sampled_negatives)}")

    return negative_samples

class Test_Dataset():
    def __init__(self, all_task_templates, dataset='steam', split='test', sample_num=1000000):
        self.all_task_templates = all_task_templates
        self.dataset = dataset  # dataset to use
        self.split = split  # train/valid/test
        try:
            self.sequential_data = ReadLineFromFile(os.path.join('./data', dataset, 'sequential_data.txt'))
        except:
            self.sequential_data = ReadLineFromFile(os.path.join('./data', dataset, 'user_sequence.txt'))
        try:
            ignore_ids_path = os.path.join('./data', dataset, 'ignore_ids.json')
            with open(ignore_ids_path, 'r') as file:
                self.ignore_ids = json.load(file)
            print(f"Successfully loaded ignore_ids: {len(ignore_ids)} items.")
        except Exception as e:
            self.ignore_ids = None
            
        try:    
            self.negative_samples = ReadLineFromFile(os.path.join('./data', dataset, 'negative_samples.txt'))
        except:
            self.negative_samples = negative_samples_maker(self.sequential_data, sample_num)
        self.test_items = None
        if os.path.exists(os.path.join('./data', dataset, 'item_datasets.pkl')):
            self.test_items = load_pickle(os.path.join('./data', dataset, 'item_datasets.pkl'))['test']
        self.meta_data = ['padding']  # item_id start from 0
        self.raw_id2meta_id = {}
        try:
            for meta in parse(os.path.join('./data', dataset, 'metadata.json')):
                self.meta_data.append(meta)
                try:
                    meta["app_name"] = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', meta["app_name"])
                except:
                    meta["app_name"] = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', meta["Game Title"])
                try:
                    self.raw_id2meta_id[meta["id"]] = len(self.meta_data) - 1
                except:
                    self.raw_id2meta_id[meta["Game ID"]] = len(self.meta_data) - 1
        except:
            for meta in parse(os.path.join('./data', dataset, 'metadata.jsonl')):
                self.meta_data.append(meta)
                try:
                    meta["app_name"] = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', meta["app_name"])
                except:
                    meta["app_name"] = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', meta["Game Title"])
                try:
                    self.raw_id2meta_id[meta["id"]] = len(self.meta_data) - 1
                except:
                    self.raw_id2meta_id[meta["Game ID"]] = len(self.meta_data) - 1
        self.search_data = None
        if os.path.exists(os.path.join('./data', dataset, 'search_data.csv')):
            self.search_data = pd.read_csv(os.path.join('./data', dataset, 'search_data.csv'))

    def gen_retrieval_data(self, sample_num):
        datasets = []
        sample_num = min(sample_num, len(self.sequential_data))
        for idx in range(sample_num):
            retrieval_datum = self.sequential_data[idx]
            sequence = [int(x) for x in retrieval_datum.split()]
            if self.ignore_ids:
                if sequence[0] in self.ignore_ids:
                    continue
            click_history = sequence[1:-1]
            target_item = sequence[-1]
            history_titles = []
            for item_id in click_history:
                item_datum = self.meta_data[item_id]
                item_title = 'unknown title'
                if 'app_name' in item_datum:
                    item_title = item_datum['app_name']
                elif 'Game Title' in item_datum:
                    item_title = item_datum['Game Title']
                history_titles.append(item_title)
            
            target_item_datum = self.meta_data[target_item]
            target_item_title = 'unknown title'
            if 'app_name' in target_item_datum:
                target_item_title = target_item_datum['app_name']
            elif 'Game Title' in target_item_datum:
                target_item_title = target_item_datum['Game Title']

            task_template = self.all_task_templates['retrieval']
            source_text = task_template['source'].format(', '.join(history_titles))
            target_text = task_template['target'].format(target_item_title)
            source_text = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', source_text)
            target_text = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', target_text)
            datasets.append({
                "source": source_text,
                "target": target_text,
                "history": history_titles,
                "task": "retrieval"
            })
        return datasets

    def gen_ranking_data(self, sample_num):
        datasets = []
        sample_num = min(sample_num, len(self.sequential_data))
        for idx in range(sample_num):
            ranking_datum = self.sequential_data[idx]
            sequence = [int(x) for x in ranking_datum.split()]            
            if self.ignore_ids:
                if sequence[0] in self.ignore_ids:
                    continue
            click_history = sequence[1:-1]
            target_item = sequence[-1]
            history_titles = []
            for item_id in click_history:
                item_datum = self.meta_data[item_id]
                item_title = 'unknown title'
                if 'app_name' in item_datum:
                    item_title = item_datum['app_name']
                elif 'Game Title' in item_datum:
                    item_title = item_datum['Game Title']
                history_titles.append(item_title)
            
            target_item_datum = self.meta_data[target_item]
            target_item_title = 'unknown title'
            if 'app_name' in target_item_datum:
                target_item_title = target_item_datum['app_name']
            elif 'Game Title' in target_item_datum:
                target_item_title = target_item_datum['Game Title']

            user_id = sequence[0]
            assert user_id == int(self.negative_samples[int(user_id)-1].split(' ', 1)[0])
            candidate_samples = self.negative_samples[int(user_id)-1].split(' ', 1)[1].split(' ')
            candidate_samples = random.sample(candidate_samples, 20)
            candidate_samples.append(str(target_item))
            random.shuffle(candidate_samples)

            candidate_titles = []
            for item_id in candidate_samples:
                item_datum = self.meta_data[int(item_id)]
                item_title = 'unknown title'
                if 'app_name' in item_datum:
                    item_title = item_datum['app_name']
                elif 'Game Title' in item_datum:
                    item_title = item_datum['Game Title']
                candidate_titles.append(item_title)

            task_template = self.all_task_templates['ranking']
            source_text = task_template['source'].format(', '.join(history_titles), ', '.join(candidate_titles))
            target_text = task_template['target'].format(target_item_title)
            source_text = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', source_text)
            target_text = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', target_text)
            datasets.append({
                "source": source_text,
                "target": target_text,
                "history": history_titles,
                "candidate": candidate_titles,
                "task": "ranking"
            })
        return datasets

    def gen_explanation_data(self, sample_num):
        datasets = []
        sample_num = min(sample_num, len(self.sequential_data))
        for idx in range(sample_num):
            exp_datum = self.sequential_data[idx]
            sequence = [int(x) for x in exp_datum.split()]
            click_history = sequence[1:-1]
            target_item = sequence[-1]
            history_titles = []
            for item_id in click_history:
                item_datum = self.meta_data[item_id]
                item_title = 'unknown title'
                if 'app_name' in item_datum:
                    item_title = item_datum['app_name']
                history_titles.append(item_title)
            
            target_item_datum = self.meta_data[target_item]
            target_item_title = 'unknown title'
            if 'app_name' in target_item_datum:
                target_item_title = target_item_datum['app_name']

            task_template = self.all_task_templates['explanation']
            source_text = task_template['source'].format(', '.join(history_titles), target_item_title)
            target_text = task_template['target'].format("No ground truth.")
            source_text = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', source_text)
            target_text = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', target_text)
            history_titles = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', ', '.join(history_titles))
            target_item_title = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', target_item_title)
            datasets.append({
                "source": source_text,
                "history": history_titles,
                "target": target_item_title,
                "task": "explanation"
            })
        return datasets
    def gen_conversation_data(self, sample_num):
        datasets = []
        sample_num = min(sample_num, len(self.sequential_data))
        for idx in range(sample_num):
            exp_datum = self.sequential_data[idx]
            sequence = [int(x) for x in exp_datum.split()]
            click_history = sequence[1:-1]
            target_item = sequence[-1]
            history_titles = []
            for item_id in click_history:
                item_datum = self.meta_data[item_id]
                item_title = 'unknown title'
                if 'app_name' in item_datum:
                    item_title = item_datum['app_name']
                history_titles.append(item_title)
            
            target_item_datum = self.meta_data[target_item]
            target_item_title = 'unknown title'
            if 'app_name' in target_item_datum:
                target_item_title = target_item_datum['app_name']

            history_titles = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', ', '.join(history_titles))
            target_item_title = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', target_item_title)
            datasets.append({
                "history": history_titles,
                "target": target_item_title,
                "task": "conversation"
            })
        return datasets

    def gen_search_data(self, sample_num):
        datasets = []
        if self.search_data is None:
            return datasets

        sample_num = min(sample_num, len(self.search_data))
        for idx in range(sample_num):
            target_item_title = self.search_data['target'][idx]
            response = self.search_data['response'][idx]
            queries = response.strip().split(",")
            test_query = random.sample(queries, 1)[0]

            task_template = self.all_task_templates['search']
            source_text = task_template['source'].format(test_query)
            target_text = task_template['target'].format(target_item_title)
            source_text = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', source_text)
            target_text = re.sub('[^A-Za-z0-9_.,!?;:\n ]', '', target_text)
            datasets.append({
                "source": source_text,
                "target": target_text,
                "task": "searching"
            })
        return datasets    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, default='ranking,retrieval,explanation,conversation', help='tasks for data generation.')
    parser.add_argument('--sample_num', type=int, default=1000, help='sample number for each task.')  
    parser.add_argument('--dataset', type=str, default='steam', help='the dataset to be evaluated, steam/beauty/sports')
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/val/test)")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    if args.dataset == 'steam':    
        from all_steam_templates import all_tasks as all_task_templates
    
    dataset = Test_Dataset(all_task_templates, dataset=args.dataset, split=args.split, sample_num=args.sample_num)

    if "retrieval" in args.tasks:
        print(f'generating retrieval data, sample number: {args.sample_num} ...')
        data = dataset.gen_retrieval_data(args.sample_num)
        fd = open(f"data/{args.dataset}/retrieval.jsonl", "w")
        for line in data:
            line = {
                "prompt": line["source"],
                "target": line["target"],
                "history": line["history"],
                "task": line["task"],
            }
            fd.write(json.dumps(line)+'\n')
    if "ranking" in args.tasks:
        print(f'generating ranking data, sample number: {args.sample_num} ...')
        data = dataset.gen_ranking_data(args.sample_num)
        fd = open(f"data/{args.dataset}/ranking.jsonl", "w")
        for line in data:
            line = {
                "prompt": line["source"],
                "target": line["target"],
                "history": line["history"],
                "candidate": line["candidate"],
                "task": line["task"],
            }
            fd.write(json.dumps(line)+'\n')
    if "explanation" in args.tasks:
        print(f'generating explanation data, sample number: {args.sample_num} ...')
        data = dataset.gen_explanation_data(args.sample_num)
        fd = open(f"data/{args.dataset}/explanation.jsonl", "w")
        for line in data:
            line = {
                "prompt": line["source"],
                "history": line["history"],
                "target": line["target"],
                "task": line["task"],
            }
            fd.write(json.dumps(line)+'\n')
    if "conversation" in args.tasks:
        print(f'generating conversation data, sample number: {args.sample_num} ...')
        data = dataset.gen_conversation_data(args.sample_num)
        fd = open(f"data/{args.dataset}/conversation.jsonl", "w")
        for line in data:
            line = {
                "history": line["history"],
                "target": line["target"],
                "task": line["task"],
                "user_simulator_system_prompt": "You are a user chatting with a recommender for recommendation in turn. Your history is {history}. Your target items: {target}.\nYou must follow the instructions below during chat.\nIf the recommender recommends {target}, you should accept.\nIf the recommender recommends other items, you should refuse them and provide the information about {target}. You should never directly tell the target item title.\nIf the recommender asks for your preference, you should provide the information about {target}. You should never directly tell the target item title.\nNow lets start, you first, act as a user.\n Your output is only allowed to be the words from the user you act.".format(
                    history=line["history"],
                    target=line["target"],
                )
            }
            fd.write(json.dumps(line)+'\n')
    if "chatbot" in args.tasks:  # use previous 3 tasks' data
        print(f'generating chatbot data, sample number: {args.sample_num} ...')

        data1 = dataset.gen_explanation_data(args.sample_num // 3)
        data2 = dataset.gen_ranking_data(args.sample_num // 3)
        data3 = dataset.gen_retrieval_data(args.sample_num // 3)

        combined_data = data1 + data2 + data3
        if len(combined_data) > args.sample_num:
            combined_data = combined_data[:args.sample_num]  
        elif len(combined_data) < args.sample_num:
            additional_data = dataset.gen_explanation_data(args.sample_num - len(combined_data))
            combined_data += additional_data

        fd = open(f"data/{args.dataset}/chatbot.jsonl", "w")
        for line in combined_data:
            line_to_write = {
                "prompt": line["source"],
                "task": "chatbot",
            }
            fd.write(json.dumps(line_to_write) + '\n')
        fd.close()

        print(f"Successfully generated {len(combined_data)} samples for chatbot.")