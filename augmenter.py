import torch
from typing import List, Tuple, Dict, Any

from .brightness import adjust_brightness_3d
from .contrast import adjust_contrast_3d
from .gaussian_noise import add_gaussian_noise_3d
from .rotation import rotate_3d
from .zoom import zoom_3d
from .flip import flip_3d

class Augmenter:
    def __init__(self, augmentations: List[Dict[str, Any]]):
        self.augmentations = augmentations

    def apply(self, image_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        
        augmented_images = []

        for image_tensor in image_tensors:
            augmented_image = image_tensor.clone()
            for aug in self.augmentations:
                aug_type = aug['type']
                params = aug['params']
                if aug_type == 'brightness':
                    augmented_image = adjust_brightness_3d(augmented_image, **params)
                elif aug_type == 'contrast':
                    augmented_image = adjust_contrast_3d(augmented_image, **params)
                elif aug_type == 'gaussian_noise':
                    augmented_image = add_gaussian_noise_3d(augmented_image, **params)
                elif aug_type == 'rotation':
                    augmented_image = rotate_3d(augmented_image, **params)
                elif aug_type == 'zoom':
                    augmented_image = zoom_3d(augmented_image, **params)
                elif aug_type == 'flip':
                    augmented_image = flip_3d(augmented_image, **params)
            augmented_images.append(augmented_image)

        return augmented_images
