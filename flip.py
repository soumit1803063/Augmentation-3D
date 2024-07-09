import torch
from typing import Tuple
from utilis import get_gpu_tensor

def flip_3d(image_tensor: torch.Tensor, axes: Tuple[bool, bool, bool]) -> torch.Tensor:

    device,image_tensor = get_gpu_tensor(image_tensor)

    if axes[0]:
        image_tensor = image_tensor.flip(0)
    if axes[1]:
        image_tensor = image_tensor.flip(1)
    if axes[2]:
        image_tensor = image_tensor.flip(2)

    return image_tensor


