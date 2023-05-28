# Benchmarking 2 ways of unpickling PyTorch tensor safely

## How to run

```
pip install -r requirements.txt

# generate pickle data to be unpickled in benchmark
python3 gen_data.py

# run the benchmark
./run_benchmarks.sh
```

The result is saved as `all.csv`

## More information

We are benchmarking three functions:

- `loads` is the standard `pickle.loads` function. This method will fail when it try to unpickle a gpu torch tensor in cpu-only environment and raise a `RuntimeError`
- `_fix_torch_loads` will detect if cuda is available, if no then call `torch.loads(..., map_location="cpu")`. The disadvantage of this method is that it may unpickle the tensor on other device (TPU?, apple silicon GPU?) to cpu even though it can be recovered to the device correctly. That's why [this pr](https://github.com/pytorch/pytorch/pull/49920) is not merged by PyTorch people
- `loads_or_fix_torch` will call vanilla `pickle.loads` by default, then catching `RuntimeError` that will be produced if gpu torch tensor is unpickled in cpu-only environment and fall back to `_fix_torch_loads`. This method is safer then call `_fix_torch_loads` directly, but catching the raised `RuntimeError` will have performance cost. This benchmark will try to measure the performance cost.

We prepare several pickled data with different sizes and if it's on GPU device. Three size of data is:

- large: 2500x2500 tensor
- medium: 500x500 tensor
- small: 5x5 tensor

We also prepare pickled numpy Ndarray data with the same sizes for comparison.

One sample results is available at <https://docs.google.com/spreadsheets/d/17i4fXPpcamTIqzg_mMBRnyY7FzQIuXiQc7dTp_iRCkA/edit?usp=sharing>
