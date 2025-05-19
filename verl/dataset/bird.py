# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the BIRD dataset to parquet format
"""

import re
import os
import datasets
import sys
sys.path.append('/home/u2020201469/NL2SQL/CTE_reasoner/SLM_module/post-training/verl/')
from verl.utils.hdfs_io import copy, makedirs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', default='./data/bird/train.json')
    parser.add_argument('--test_data_dir', default='./data/bird/dev.json')
    parser.add_argument('--local_dir', default='./dataset/full_set')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_name = 'bird'
    
    # 从本地data source读入完整数据集
    train_dataset = datasets.load_dataset(
        "json",
        data_files=os.path.join(args.train_data_dir),
        split="train"
    )
    test_dataset = datasets.load_dataset(
        "json",
        data_files=os.path.join(args.test_data_dir),
        split="train"
    )
    # 过滤test_dataset中prompt长度超过8000的项
    test_dataset = test_dataset.filter(lambda x: len(x["instruction"]) <= 10000)
    train_dataset = train_dataset.filter(lambda x: len(x["instruction"]) <= 11000)
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            if "prompt" in example:
                question_raw = example.pop('prompt')
            else:
                question_raw = example.pop('instruction')

            question = question_raw
            answer_raw = example.pop('output')
            data = {
                "data_source": data_name,
                "prompt": question,
                "ability": "sql",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer_raw
                },
                "extra_info": {
                    'split': split,
                    'idx': example['id'] if 'id' in example else idx,
                    'answer': answer_raw,
                    "question": question_raw,
                    "db_id": example['db_id'],
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    # 统计train_dataset最长prompt的长度
    print(f"train_dataset length: {len(train_dataset)}")
    print(f"test_dataset length: {len(test_dataset)}")
    max_prompt_length = max(len(item['prompt']) for item in train_dataset)
    print(f"最长prompt长度: {max_prompt_length}")
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, f'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, f'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
