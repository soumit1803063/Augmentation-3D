import torch
from utilis import get_gpu_tensor

def adjust_contrast_3d(image_tensor: torch.Tensor, factor: float) -> torch.Tensor:
    device,image_tensor = get_gpu_tensor(image_tensor)
    mean = torch.mean(image_tensor)
    contrast_tensor = (image_tensor - mean) * factor + mean

    return contrast_tensor
