# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import random
import math
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import argparse

from utils import get_item_text
from template import querysummary2item_template, title2item_template

random.seed(2023)

def parse_args():
    parser = argparse.ArgumentParser(description="merge")
    parser.add_argument(
        "--in_seq_data", type=str, help=""
    )
    parser.add_argument(
        "--in_meta_data", type=str, help=""
    )
    parser.add_argument(
        "--in_u2i", type=str, help=""
    )
    parser.add_argument(
        "--in_q2i", type=str, help=""
    )
    parser.add_argument(
        "--in_q2i_misspell", type=str, help=""
    )
    parser.add_argument(
        "--gpt_path", type=str, help=""
    )
    parser.add_argument(
        "--out_gpt", type=str, help=""
    )
    parser.add_argument(
        "--neg_num", type=int, default=7, help=""
    )
    args = parser.parse_args()
    return args

def merge_and_sample(itemid2text, itemid2title, itemid2features, args):
    query_profile = pd.read_csv(args.gpt_path + '.csv', header=None, sep=',', names=['question', 'target'])
    query_profile = query_profile.iloc[1:].reset_index(drop=True)  # Reset index after slicing

    idx = 0
    id2queries = defaultdict(list)
    uidiid2query = {}

    with open(args.out_gpt, 'w') as w:
        count = 0
        
        # Handle user-to-item (u2i) interactions
        with open(args.in_u2i, 'r') as f:
            for line in tqdm(f):
                line = json.loads(line)
                userid = int(line['userid'])
                target_item = int(line['target_id'])

                if idx >= len(query_profile):
                    print(f"Reached the end of query profile at index {idx}.")
                    break  # Avoid out-of-bounds error

                query = query_profile.loc[idx, 'target']
                uidiid2query[(userid, target_item)] = query
                neg_items = []

                while len(neg_items) < args.neg_num:
                    neg_item = random.randint(1, len(itemid2title) - 1)
                    if neg_item != target_item:
                        neg_items.append(neg_item)

                output = {
                    'query': query,
                    'pos': [itemid2text[target_item]],
                    'neg': [itemid2text[x] for x in neg_items]
                }
                w.write(json.dumps(output) + '\n')
                count += 1
                idx += 1  # Increment the index after processing each line

        print("u2i_gpt4 count:", count)

        # Reset index for query-profile in q2i part
        idx = 0
        count = 0
        
        # Handle query-to-item (q2i) interactions
        with open(args.in_q2i, 'r') as f:
            for line in tqdm(f):
                line = json.loads(line)
                target_item = int(line['item_id'])

                if idx >= len(query_profile):
                    print(f"Reached the end of query profile at index {idx}.")
                    break

                query = query_profile.loc[idx, 'target'].split('#SEP#')
                id2queries[target_item] = query

                for q in query:
                    neg_items = []
                    while len(neg_items) < args.neg_num:
                        neg_item = random.randint(1, len(itemid2title) - 1)
                        if neg_item != target_item:
                            neg_items.append(neg_item)

                    output = {
                        'query': q,
                        'pos': [itemid2text[target_item]],
                        'neg': [itemid2text[x] for x in neg_items]
                    }
                    w.write(json.dumps(output) + '\n')
                    count += 1

                idx += 1

        print("q2i_gpt4 count: ", count)
        print('idx: ', idx)
        count = 0
        with open(args.in_q2i_misspell, 'r') as f:
            idx = 0
            for line in tqdm(f):
                line = json.loads(line)
                target_item = int(line['item_id'])
                query = query_profile.loc[idx, 'target']
                query = query.split('#SEP#')
                for q in query:
                    if random.random() < 0.5:
                        template = "{}"
                    else:
                        template = random.choice(title2item_template)
                    q = template.format(q)
                    neg_items = []
                    while len(neg_items) < args.neg_num:
                        neg_item = random.randint(1, len(itemid2title)-1)
                        if neg_item != target_item and itemid2title[target_item][1]!=itemid2title[neg_item][1]:
                            neg_items.append(neg_item)
                    output = {
                        'query': q,
                        'pos': [itemid2text[target_item]],
                        'neg': [itemid2text[x] for x in neg_items]
                    }
                    w.write(json.dumps(output) + '\n')
                    count+=1
                idx += 1
        print("q2i_misspell_gpt4 count: ", count)
        print('idx: ', idx)
        idx = 0
        count = 0
        for (userid, target_item), user_hist in tqdm(uidiid2query.items()):
            user_queries = random.sample(id2queries[target_item], min(2, len(id2queries[target_item])))

            for user_query in user_queries:
                template = random.choice(querysummary2item_template)
                query = template.format(user_hist, user_query)
                neg_items = []
                while len(neg_items) < args.neg_num:
                    neg_item = random.randint(1, len(itemid2title)-1)
                    if neg_item != target_item:
                        neg_items.append(neg_item)
                output = {
                    'query': query,
                    'pos': [itemid2text[target_item]],
                    'neg': [itemid2text[x] for x in neg_items]
                }
                w.write(json.dumps(output) + '\n')
                count+=1
        print("gpt_querysummary count: ", count)

    
if __name__ == "__main__":
    args = parse_args()
    itemid2text, itemid2title, itemid2features, _ = get_item_text(args.in_meta_data)
    merge_and_sample(itemid2text, itemid2title, itemid2features, args)