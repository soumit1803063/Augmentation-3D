import torch
import torch.nn.functional as F
from typing import Tuple
from utilis import get_gpu_tensor

def rotate_3d(image_tensor: torch.Tensor, angle: float, axis: Tuple[int, int, int] = (0, 1, 0)) -> torch.Tensor:

    # Move the tensor to the GPU
    device,image_tensor = get_gpu_tensor(image_tensor)

    # Compute the rotation matrix
    angle_rad = torch.tensor(angle * (torch.pi / 180), device=device)
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    
    axis = torch.tensor(axis, device=device, dtype=torch.float32)
    axis = axis / torch.sqrt(torch.sum(axis ** 2))
    
    R = torch.eye(4, device=device)
    R[:3, :3] = cos_a * torch.eye(3, device=device) + (1 - cos_a) * torch.outer(axis, axis) + sin_a * torch.tensor([[0, -axis[2], axis[1]],
                                                                                                                     [axis[2], 0, -axis[0]],
                                                                                                                     [-axis[1], axis[0], 0]], device=device)
    
    # Apply rotation using affine_grid and grid_sample
    D, H, W = image_tensor.shape
    grid = F.affine_grid(R[:3, :4].unsqueeze(0), [1, 1, D, H, W], align_corners=False)
    rotated_tensor = F.grid_sample(image_tensor.unsqueeze(0).unsqueeze(0), grid, mode='bilinear', align_corners=False)

    return rotated_tensor.squeeze(0).squeeze(0)

