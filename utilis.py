import torch

def get_gpu_tensor(image_tensor: torch.Tensor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = image_tensor.to(device)
    return device,image_tensor