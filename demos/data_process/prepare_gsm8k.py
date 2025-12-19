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
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets
from datasets import Dataset
from verl.utils.hdfs_io import copy, makedirs

def makeUpData():
    
    user_prompt = """
    # 任务定义
    用户和助手之间的对话。用户提出一个问题，助手给出回答

    # 响应格式
    Let\'s think step by step and output the final answer after "####".

    # 用户问题
    """

    train_data = {
        "question": [
            f"{user_prompt}\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            f"{user_prompt}\nWeng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
            f"{user_prompt}\nBetty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
            f"{user_prompt}\nJulie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the  remaining pages tomorrow, how many pages should she read?",
            f"{user_prompt}\nJames writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"],
        "answer": [
            """Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72""",
            """Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10""",
            """In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30. This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. #### 5""",
            """Maila read 12 x 2 = <<12*2=24>>24 pages today. So she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday. There are 120 - 36 = <<120-36=84>>84 pages left to be read.Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages. #### 42""",
            """He writes each friend 3*2=<<3*2=6>>6 pages a week So he writes 6*2=<<6*2=12>>12 pages every week That means he writes 12*52=<<12*52=624>>624 pages a year #### 624"""]
    }
    test_data = {
        "question": [f"{user_prompt}\nJanet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                     f"{user_prompt}\nA robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                     f"{user_prompt}\nJosh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?"],
        "answer": [
            """Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
#### 18""",
            """It takes 2/2=<<2/2=1>>1 bolt of white fiber
So the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric
#### 3""",
            """The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000
He increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000
So the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000
So he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000
#### 70000"""]
    }

    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)

    ds = datasets.DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    return ds

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "openai/gsm8k"
    dataset=makeUpData()
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)
