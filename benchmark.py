import argparse
import os
import glob
import time
import pickle

import pandas as pd

from torch_unpickler import _fix_torch_loads, loads_or_fix_torch


def load_bss():

    paths = glob.glob("data/*.pickle")
    paths.sort()

    d = {}

    for path in paths:
        k = path.split(".")[0]
        with open(path, "rb") as f:
            d[k] = f.read()

    return d


def benchmark(bs, bs_name, unpickling_f, repeat):
    start = time.time()
    for _ in range(repeat):
        tmp = unpickling_f(bs)
    end = time.time()
    total_time = end - start
    avg_time = total_time / repeat
    function_name = unpickling_f.__name__
    print(f"{function_name} loop {repeat} times for {bs_name}, spend total time {total_time} seconds, average {avg_time} seconds per loop")
    return [function_name, bs_name, repeat, total_time, avg_time]
    


if __name__ == "__main__":

    function_map = {}
    for func in _fix_torch_loads, loads_or_fix_torch, pickle.loads:
        function_map[func.__name__] = func


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disable-gpu", help="disable GPU using CUDA_VISIBLE_DEVICES=-1",
        action="store_true"
    )
    parser.add_argument(
        "--function-name", help="specify which function to test, select from `loads`, `_fix_torch_loads` and `loads_or_fix_torch`",
    )

    args = parser.parse_args()
    gpu = not args.disable_gpu
    func_name = args.function_name
    func = function_map[func_name]

    if not gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # preload torch to save import time in benchmark function
    import torch

    bs_d = load_bss()
    base_repeat = 100000
    stats = []
    for bs_name, bs in bs_d.items():
        if "small" in bs_name:
            repeat = base_repeat * 100
        elif "medium" in bs_name:
            repeat = base_repeat * 10
        else:
            repeat = base_repeat

        # plain pickle.loads only works with gpu when loading gpu tensors
        if func_name == "loads" and not gpu and "gpu" in bs_name:
            row = ["loads", bs_name, None, None, None]
        else:
            row = benchmark(bs, bs_name, func, repeat=repeat)

        row.append(gpu)
        stats.append(row)
        torch.cuda.empty_cache()

    df = pd.DataFrame(stats, columns=["function", "data", "repeat", "total time", "avg time", "cuda availablity"])
    path = f"{func_name}_gpu.csv" if gpu else f"{func_name}_nogpu.csv"
    path = "outputs/" + path
    df.to_csv(path, index=False, header=False)

