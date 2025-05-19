import numpy as np
import torch

def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

def numpy2torch(numpy_obj):
    torch_obj = torch.tensor(numpy_obj)
    return 2 * (torch_obj / 255) - 1


