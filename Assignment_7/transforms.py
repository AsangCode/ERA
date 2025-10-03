import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class Transforms:
    def __init__(self, mean, std):
        self.train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.CoarseDropout(
                max_holes=1, max_height=16, max_width=16,
                min_holes=1, min_height=16, min_width=16,
                fill_value=mean, mask_fill_value=None,
                p=0.5
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

        self.test_transforms = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
