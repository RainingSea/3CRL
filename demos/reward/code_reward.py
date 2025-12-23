import base64
import json
import multiprocessing
import pickle
import zlib
import wandb

# Reuse `run_test` for convenience
# from verl.utils.reward_score.prime_code.testing_util_old import run_test
from verl.utils.reward_score.prime_code.testing_util import run_test
from verl.utils.tracking import Tracking

def _temp_run(in_outs, generation, debug, result, metadata_list, timeout):
    """
    这是一个 多进程目标函数
    调用 run_test 执行生成的代码，并将结果存入共享列表（manager.list()）
    之所以用多进程，是为了能强制终止失控的代码（如死循环、无限递归）
    """
    res, metadata = run_test(in_outs, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)

def check_correctness(in_outs, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`
    使用 multiprocessing.Manager() 创建共享变量 result 和 metadata_list
    启动子进程运行 _temp_run
    """

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(in_outs, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=(timeout + 1) * len(in_outs["inputs"]) + 5)
    if p.is_alive():
        p.kill()
    if not result:
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print("global timeout")
    return result[0], metadata_list[0]


import ray
def compute_score_livecodebench(completion, test_cases):
    compute_result = compute_score_livecodebench_impl(completion, test_cases)
    return compute_result

wandb.init()

def compute_score_livecodebench_impl(completion, test_cases):
    solution = completion.split("```python")[-1].split("```")[0]
    wandb.log({"solution": test_cases})
    # logger.log("Evaluating solution:\n"+solution)
    print("1 - 测试用例为 "+str(test_cases))
    # extract test cases
    if type(test_cases) != dict:
        try:
            # 正常情况下应该解析出一个字典，{"inputs": [c1,c2,c3], "outputs": [r1,r2,r3]}
            in_outs = json.loads(test_cases)
        except Exception as e:
            # 有可能压缩过，需要解压
            try:
                # 正常情况下应该解析出一个字典，{"inputs": [c1,c2,c3], "outputs": [r1,r2,r3]}
                print("may has zip, try to unzip")
                in_outs = json.loads(pickle.loads(zlib.decompress(base64.b64decode(test_cases.encode("utf-8")))))
                # print("测试用例为 "+str(in_outs))
            except Exception as e:
                print(f"After unzip still Error: {e}")
                return False
    else:
        in_outs = test_cases
        
    success = False
    try:
        res, metadata = check_correctness(in_outs=in_outs, generation=solution, timeout=6, debug=False)
        print(f"生成代码:{solution}\n测试结果：{str(res)}\n测试元数据：{str(metadata)}")

        # success = all(map(lambda x: x is True, res))
        success = res.count(True) / len(in_outs["inputs"])  # 允许部分测试用例失败
    except Exception:
        pass

    return success

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    接收模型生成的代码，并且跑测试用例来进行测试
    """
    
    result = compute_score_livecodebench(solution_str, ground_truth)
    print("奖励为："+str(result))
    
    if result:
        return 1.0
    else:
        return 0.0
    
    # res, metadata = run_test(sample=example_io, test=solution_str, debug=False, timeout=60)
    # return sum(res)/len(res)

gencode1="""from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        INF = 10**18
        
        # dp[a][b] = minimum length with starting char a and ending char b
        dp = [[INF] * 26 for _ in range(26)]
        
        first_word = words[0]
        f = ord(first_word[0]) - ord('a')
        l = ord(first_word[-1]) - ord('a')
        dp[f][l] = len(first_word)
        
        for w in words[1:]:
            nf = ord(w[0]) - ord('a')
            nl = ord(w[-1]) - ord('a')
            wlen = len(w)
            
            new_dp = [[INF] * 26 for _ in range(26)]
            
            for a in range(26):
                for b in range(26):
                    if dp[a][b] == INF:
                        continue
                    
                    cur = dp[a][b]
                    
                    # join(current, w)
                    cost1 = cur + wlen - (1 if b == nf else 0)
                    new_dp[a][nl] = min(new_dp[a][nl], cost1)
                    
                    # join(w, current)
                    cost2 = cur + wlen - (1 if nl == a else 0)
                    new_dp[nf][b] = min(new_dp[nf][b], cost2)
            
            dp = new_dp
        
        return min(min(row) for row in dp)"""

gencode2="""def main():
    import sys
    data = sys.stdin.read().splitlines()
    t = int(data[0])
    
    for i in range(1, t + 1):
        s = data[i].strip()
        # Split the string into contiguous groups of same characters
        groups = []
        current_char = s[0]
        count = 1
        for char in s[1:]:
            if char == current_char:
                count += 1
            else:
                groups.append((current_char, count))
                current_char = char
                count = 1
        groups.append((current_char, count))
        
        # Extract lengths of groups that are '1's
        ones_lengths = [length for char, length in groups if char == '1']
        # Sort in descending order: players will pick largest available group of 1s first
        ones_lengths.sort(reverse=True)
        
        # Alice picks first, then Bob, then Alice, etc.
        # So Alice gets the 0th, 2nd, 4th, ... elements (even indices)
        alice_score = sum(ones_lengths[j] for j in range(0, len(ones_lengths), 2))
        print(alice_score)

if __name__ == "__main__":
    main()"""

if __name__=="__main__":
    example_io1={
        "inputs":[
          "5\n01111001\n0000\n111111\n101010101\n011011110111\n",
          
        ],
        "outputs":[
          "4\n0\n6\n3\n6\n"
        ]
      }
    example_io2={
        "inputs":[
          "5\n01111001\n0000\n111111\n101010101\n011011110111\n",
          "5\n01111001\n0000\n111111\n101010101\n011011110111\n"
        ],
        "outputs":[
          "4\n0\n6\n3\n6\n",
          "4\n0\n6\n3\n7\n"
        ],
      }
    
    res, metadata = run_test(sample=example_io2, test=gencode2, debug=False, timeout=60)
    print(res)