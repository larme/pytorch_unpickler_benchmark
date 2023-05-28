# run with cuda
import pickle

import numpy as np
import torch


def make_dict(tensor):
    return dict(a=1, tensor=tensor)


sizes = {
    "large": (2500, 2500),
    "medium": (500, 500),
    "small": (5, 5),
}


if __name__ == "__main__":
    for size in sizes.keys():
        tensor = torch.rand(sizes[size])
        d = make_dict(tensor)
        with open(f"data/{size}_dict.pickle", "wb") as f:
            pickle.dump(d, f)

        ndarray = tensor.numpy()
        d = make_dict(ndarray)
        with open(f"data/np_{size}_dict.pickle", "wb") as f:
            pickle.dump(d, f)

        tensor_on_gpu = tensor.to("cuda")
        d_on_gpu = make_dict(tensor_on_gpu)
        with open(f"data/{size}_dict_on_gpu.pickle", "wb") as f:
            pickle.dump(d_on_gpu, f)
        
