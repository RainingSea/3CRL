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
Preprocess the dataset to parquet format
"""

import argparse
import os
from functools import partial

from datasets import concatenate_datasets, load_dataset

from verl.utils.hdfs_io import copy, makedirs


def example_map_fn(example, idx, process_fn, data_source, ability, split):
    """
    将数据集处理成verl需要的格式
    """
    question, solution = process_fn(example)
    data = {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {"split": split, "index": idx},
    }
    return data


def build_aime2024_dataset():
    def process_aime2024(example):
        return example["Problem"], str(example["Answer"])

    data_source = "Maxwell-Jia/AIME_2024"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = load_dataset(data_source, split="train")
    map_fn = partial(
        example_map_fn, process_fn=process_aime2024, data_source=data_source, ability="English", split="test"
    )
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_gpqa_dimond_dataset():
    import random

    GPQA_QUERY_TEMPLATE = (
        "Answer the following multiple choice question. The last line of your response should be of the following "
        "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before "
        "answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    )

    def process_gpqa_diamond(example):
        choices = [example["Incorrect Answer 1"], example["Incorrect Answer 2"], example["Incorrect Answer 3"]]
        random.shuffle(choices)
        gold_index = random.randint(0, 3)
        choices.insert(gold_index, example["Correct Answer"])
        query_prompt = GPQA_QUERY_TEMPLATE.format(
            A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=example["Question"]
        )
        gold_choice = "ABCD"[gold_index]
        return query_prompt, gold_choice

    data_source = "Idavidrein/gpqa"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    dataset = load_dataset(data_source, "gpqa_diamond", split="train")
    map_fn = partial(
        example_map_fn, process_fn=process_gpqa_diamond, data_source=data_source, ability="Math", split="test"
    )
    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    return dataset


def build_cnmo2024_dataset():
    def process_cnmo2024(example):
        return example["question"], example["answer"]

    data_source = "opencompass/LiveMathBench"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    dataset_en = load_dataset(data_source, "v202412_CNMO_en", split="test")
    map_fn_en = partial(
        example_map_fn, process_fn=process_cnmo2024, data_source="opencompass/cnmo2024_en", ability="Math", split="test"
    )
    dataset_en = dataset_en.map(map_fn_en, with_indices=True, remove_columns=dataset_en.column_names)

    dataset_zh = load_dataset(data_source, "v202412_CNMO_cn", split="test")
    map_fn_zh = partial(
        example_map_fn, process_fn=process_cnmo2024, data_source="opencompass/cnmo2024_zh", ability="Math", split="test"
    )
    dataset_zh = dataset_zh.map(map_fn_zh, with_indices=True, remove_columns=dataset_zh.column_names)

    dataset = concatenate_datasets([dataset_en, dataset_zh])
    return dataset


def build_livecodebench_dataset():
    import base64
    import json
    import pickle
    import zlib

    def process_livecodebench(example):
        # Construct Query Prompt
        # From https://github.com/LiveCodeBench/LiveCodeBench/blob/998c52d394b836f15fff3b9a29866191108ff81b/lcb_runner/prompts/code_generation.py#L140
        query_prompt = (
            f"You will be given a question (problem specification) and will generate a correct Python program "
            f"that matches the specification and passes all tests.\n\nQuestion: {example['question_content']}\n\n"
        )
        if example["starter_code"]:
            query_prompt += (
                f"You will use the following starter code to write the solution to the problem and enclose your "
                f"code within delimiters.\n```python\n{example['starter_code']}\n```"
            )
        else:
            query_prompt += (
                "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test "
                "on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
                "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
                "```python\n# YOUR CODE HERE\n```"
            )

        # Construct test cases
        public_test_cases = json.loads(example["public_test_cases"])
        try:
            private_test_cases = json.loads(example["private_test_cases"])
        except Exception as e:
            private_test_cases = json.loads(
                pickle.loads(zlib.decompress(base64.b64decode(example["private_test_cases"].encode("utf-8"))))
            )
            print(f"Error loading private test cases: {e}")
        full_test_cases = public_test_cases + private_test_cases

        metadata = json.loads(example["metadata"])
        test_cases = {
            "inputs": [t["input"] for t in full_test_cases],
            "outputs": [t["output"] for t in full_test_cases],
            "fn_name": metadata.get("func_name", None),
        }
        text_cases_compressed = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(test_cases)))).decode("utf-8")
        return query_prompt, text_cases_compressed

    data_source = "livecodebench/code_generation_lite"
    print(f"Loading the {data_source} dataset from modelscope...", flush=True)
    from modelscope.msdatasets import MsDataset
    dataset =  MsDataset.load('livecodebench/code_generation_lite', version_tag="release_v2",trust_remote_code=True,split="test")
    # 只取4个数据
    
    # R1 Evaluation use LiveCodeBench 24.08-25.01
    # dataset = dataset.filter()
    map_fn = partial(
        example_map_fn, process_fn=process_livecodebench, data_source=data_source, ability="Code", split="test"
    )

    dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names, num_proc=8)
    return dataset

def build_apps_dataset(base_path=None):
    import os
    import json
    import base64
    import pickle
    import zlib
    from datasets import Dataset
    from functools import partial

    if base_path is None:
        raise ValueError("Please provide the base path for the apps dataset.")
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"apps base path not found: {base_path}")

    # Step 1: 构建原始样本列表（每个样本是一个 dict，模拟 parquet 行）
    raw_examples = []
    folders = sorted(os.listdir(base_path))
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue

        try:
            qpath = os.path.join(folder_path, "question.txt")
            iopath = os.path.join(folder_path, "input_output.json")
            metapath = os.path.join(folder_path, "metadata.json")

            with open(qpath, "r", encoding="utf-8") as f:
                question_text = f.read().strip()

            with open(iopath, "r", encoding="utf-8") as f:
                io_data = json.load(f)

            with open(metapath, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # 构造一个“原始样本”，字段名与 LiveCodeBench 对齐（便于复用 process_fn）
            
            raw_example = {
                "question_content": question_text,
                "public_test_cases": [],  # APPS 没有 public/private 区分，全部视为 public
                "private_test_cases": [],  # 我们把所有 test cases 放在 public 里，private 留空
                "starter_code": "",       # APPS 通常无 starter code
                "metadata": meta,
                "io_data": io_data,       # 自定义字段，用于 process_fn 提取 inputs/outputs
                "extra_info": {"idx":folder,"difficulty": meta.get("difficulty","unknown"),"url": meta.get("url","unknown")}
            }
            raw_examples.append(raw_example)

        except Exception as e:
            print(f"Skipping {folder}: {e}")
            continue

    if not raw_examples:
        return Dataset.from_list([])

    # Step 2: 转为 HuggingFace Dataset
    dataset = Dataset.from_list(raw_examples)

    # Step 3: 定义针对 APPS 的 process_fn（注意：它接收的是上面构造的 raw_example）
    def process_apps(example):
        # 构建 prompt
        query_prompt = (
            "You will be given a question (problem specification) and will generate a correct Python program "
            "that matches the specification and passes all tests.\n\nQuestion: {}\n\n".format(example["question_content"])
        )
        query_prompt += (
            "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test "
            "on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
            "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
            "```python\n# YOUR CODE HERE\n```"
        )

        # 构建 test cases
        io_data = example["io_data"]
        metadata = example["metadata"]
        test_cases = {
            "inputs": io_data.get("inputs", []),
            "outputs": io_data.get("outputs", []),
            "fn_name": metadata.get("fn_name") if isinstance(metadata, dict) else None,
        }

        # 压缩 ground truth
        text_cases_compressed = base64.b64encode(
            zlib.compress(pickle.dumps(json.dumps(test_cases)))
        ).decode("utf-8")

        return query_prompt, text_cases_compressed

    # Step 4: 复用统一的 map 函数
    data_source = "local_apps"
    map_fn = partial(
        example_map_fn,
        process_fn=process_apps,
        data_source=data_source,
        ability="Code",
        split="test"
    )

    dataset = dataset.map(
        map_fn,
        with_indices=True,
        remove_columns=dataset.column_names,
        num_proc=8
    )

    return dataset
    

TASK2DATA = {
    "aime2024": build_aime2024_dataset,
    "gpqa_diamond": build_gpqa_dimond_dataset,
    "cnmo2024": build_cnmo2024_dataset,
    "livecodebench": build_livecodebench_dataset,
    "apps": partial(build_apps_dataset, base_path=r"D:\Project\Datasets\APPS_DATA\train"),
}
SUPPORTED_TASKS = TASK2DATA.keys()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/r1")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--tasks", default="all")

    args = parser.parse_args()

    if args.tasks.lower() == "all":
        args.tasks = SUPPORTED_TASKS
    else:
        args.tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
        for task in args.tasks:
            if task not in SUPPORTED_TASKS:
                raise NotImplementedError(f"{task} has not been supported.")

    datasets = []
    for task in args.tasks:
        datasets.append(TASK2DATA[task]())
        
    test_dataset = concatenate_datasets(datasets)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset.to_parquet(os.path.join(local_dir, "test_apps.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
