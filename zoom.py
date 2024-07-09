import torch
from typing import Tuple
from utilis import get_gpu_tensor
def zoom_3d(image_tensor: torch.Tensor, 
            center: Tuple[int, int, int], 
            zoom_factor: float) -> torch.Tensor:

    # Move the tensor to the GPU
    device,image_tensor = get_gpu_tensor(image_tensor)

    # Get the dimensions of the image
    D, H, W = image_tensor.shape

    # Compute the size of the zoomed area
    zoomed_D = int(D / zoom_factor)
    zoomed_H = int(H / zoom_factor)
    zoomed_W = int(W / zoom_factor)

    # Get the center coordinates
    center_d, center_h, center_w = center

    # Compute the start and end indices for the zoomed area
    start_d = max(0, center_d - zoomed_D // 2)
    start_h = max(0, center_h - zoomed_H // 2)
    start_w = max(0, center_w - zoomed_W // 2)

    end_d = min(D, start_d + zoomed_D)
    end_h = min(H, start_h + zoomed_H)
    end_w = min(W, start_w + zoomed_W)

    # Extract the zoomed area
    zoomed_tensor = image_tensor[start_d:end_d, start_h:end_h, start_w:end_w]

    # Resize the zoomed tensor back to the original size
    zoomed_tensor = torch.nn.functional.interpolate(
        zoomed_tensor.unsqueeze(0).unsqueeze(0),
        size=(D, H, W),
        mode='trilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)

    return zoomed_tensor
