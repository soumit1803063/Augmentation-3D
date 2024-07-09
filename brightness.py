import torch
from utilis import get_gpu_tensor

def adjust_brightness_3d(image_tensor: torch.Tensor, factor: float) -> torch.Tensor:
    device,image_tensor = get_gpu_tensor(image_tensor)
    brightened_tensor = image_tensor * factor

    return brightened_tensor
