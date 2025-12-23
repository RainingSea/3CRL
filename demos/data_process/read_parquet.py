#!/usr/bin/env python3
"""
读取 Parquet 文件并显示基本信息。
支持本地文件路径，也可扩展支持 S3（需配置）。
"""

import argparse
import pandas as pd
import os
import pandas as pd
import base64
import zlib
import pickle
import json


import pandas as pd

def json_to_parquet(json_path, parquet_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    df.to_parquet(parquet_path, engine="pyarrow")
    

def decode_ground_truth(encoded_str: str):
    if not isinstance(encoded_str, str):
        return encoded_str
    
    compressed_bytes = base64.b64decode(encoded_str)
    pickled_bytes = zlib.decompress(compressed_bytes)
    json_str = pickle.loads(pickled_bytes)
    return json.loads(json_str)

def parquet_to_json(parquet_path, json_path):
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    
    # 正确处理嵌套字段
    def decode_reward_model(rm):
        if not isinstance(rm, dict):
            return rm

        if "ground_truth" in rm:
            rm = rm.copy()  # 防止就地修改
            test_case = rm["ground_truth"]
            if not isinstance(test_case, str):
                return rm
            try:
                in_outs = json.loads(test_case)
            except Exception as e:
                try:
                    in_outs = json.loads(pickle.loads(zlib.decompress(base64.b64decode(test_case.encode("utf-8")))))
                except Exception as e:
                    print(f"Data Error even try to unzip when loading test cases: {e}")
                    return False
            rm["ground_truth"] = in_outs
        return rm
                
    df["reward_model"] = df["reward_model"].apply(decode_reward_model)
    # 保存为 JSON
    output_path = json_path
    df.head(50).to_json(
        output_path,
        orient="records",
        force_ascii=False,
        indent=2
    )

if __name__ == "__main__":
    json_path = r"D:\Project\C3RL\demos\data\data_json\apps_10P.json"
    
    full_parquet_path = r"D:\Project\C3RL\demos\data\test_apps.parquet"
    parquet_path = r"D:\Project\C3RL\demos\data\test_4P_lite.parquet"
    
    
    json_to_parquet(json_path, parquet_path)
    # parquet_to_json(full_parquet_path, json_path)
    