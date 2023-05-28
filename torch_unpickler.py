import io
import pickle

def _safe_torch_tensor_loads(bs):
    import torch
    f = io.BytesIO(bs)
    if not torch.cuda.is_available():
        return torch.load(f, map_location="cpu")
    else:
        return torch.load(f)

class FixTorchUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return _safe_torch_tensor_loads
        else: return super().find_class(module, name)

def _fix_torch_loads(bs):
    f = io.BytesIO(bs)
    unpickler = FixTorchUnpickler(f)
    return unpickler.load()


def loads_or_fix_torch(bs):
    try:
        return pickle.loads(bs)
    except RuntimeError:
        return _fix_torch_loads(bs)
