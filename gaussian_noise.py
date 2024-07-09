import torch
from utilis import get_gpu_tensor

def add_gaussian_noise_3d(image_tensor: torch.Tensor, mean: float = 0.0, stddev: float = 1.0) -> torch.Tensor:
    device,image_tensor = get_gpu_tensor(image_tensor)
    noise = torch.randn_like(image_tensor) * stddev + mean
    noisy_tensor = image_tensor + noise

    return noisy_tensor

