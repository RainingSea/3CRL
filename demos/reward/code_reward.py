import base64
import json
import multiprocessing
import pickle
import zlib
import wandb

# Reuse `run_test` for convenience
# from verl.utils.reward_score.prime_code.testing_util_old import run_test
from verl.utils.reward_score.prime_code.testing_util import run_test

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
# @ray.remote
def compute_score_livecodebench(completion, test_cases):
    compute_result = compute_score_livecodebench_impl(completion, test_cases)
    return compute_result


def compute_score_livecodebench_impl(completion, test_cases):
    solution = completion.split("```python")[-1].split("```")[0]
    
    # extract test cases
    try:
        # 正常情况下应该解析出一个字典，{"inputs": [c1,c2,c3], "outputs": [r1,r2,r3]}
        in_outs = json.loads(test_cases)
    except Exception as e:
        # 有可能压缩过，需要解压
        print(f"Data may be zipped, Error loading test cases: {e}")
        try:
            # 正常情况下应该解析出一个字典，{"inputs": [c1,c2,c3], "outputs": [r1,r2,r3]}
            in_outs = json.loads(pickle.loads(zlib.decompress(base64.b64decode(test_cases.encode("utf-8")))))
            
            # wandb.init()
            # wandb.log({"Test Cases after unzip": in_outs})
        except Exception as e:
            print(f"After unzip still Error: {e}")
            return False

    success = False
    try:
        res, metadata = check_correctness(in_outs=in_outs, generation=solution, timeout=6, debug=False)
        print("测试结果："+str(res))
        print("测试元数据："+str(metadata))
        # wandb.log({"Reward Return": (res, metadata)})
        success = all(map(lambda x: x is True, res))
    except Exception:
        pass

    return success

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    接收模型生成的代码，并且跑测试用例来进行测试
    """
    
    result = compute_score_livecodebench(solution_str, ground_truth)
    print("蒋黎维："+str(result))
    if result:
        return 1.0
    else:
        return 0.0
    
    # res, metadata = run_test(sample=example_io, test=solution_str, debug=False, timeout=60)
    # return sum(res)/len(res)