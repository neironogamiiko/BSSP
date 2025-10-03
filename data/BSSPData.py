import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

def BSSPAugmentation():
    """
    Створює композицію аугментацій для зображень та відповідних heatmap-ів.

    Аугментації включають:
        1. Додавання гаусівського шуму з випадковим стандартним відхиленням.
        2. Випадкове змінення яскравості та контрасту.
        3. Обертання зображення на +/-5 градусів.
    """
    return A.Compose([
        A.GaussNoise(std_range=(0.0, 0.01), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.5, p=0.5),
        A.Rotate(limit=5, p=0.5),  # геометричні зміни одночасно для image і mask
    ], additional_targets={'mask': 'mask'})  # mask = heatmap

class BSSPDataset(Dataset):
    """
    Датасет для завантаження зображень і відповідних heatmap-ів з .npy файлів.

    Кожен елемент датасету - це пара:
        - зображення у форматі (C, H, W), float32
        - кортеж із трьох копій heatmap (multi-output), кожна у форматі (C, H, W), float32

    Формат даних у .npy:
        - зображення: (H, W, 3), значення float32, нормалізовані
        - heatmap:    (H, W, num_classes), float32
    """
    def __init__(self, image_path: str, heatmap_path: str, augmentation = None):
        """
        Ініціалізація даних.

        :param image_path: шлях до директорії з .npy-зображеннями
        :param heatmap_path: шлях до директорії з .npy-heatmap файлами
        """
        self.image_path = Path(image_path)
        self.heatmap_path = Path(heatmap_path)
        self.files = sorted(self.image_path.glob("*.npy"))
        self.augmentation = augmentation


    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        image_path = self.files[i]
        # вважаємо, що відповідний heatmap має те саме ім'я
        heatmap_path = self.heatmap_path / image_path.name

        image = np.load(image_path).astype(np.float32)
        heatmap = np.load(heatmap_path).astype(np.float32)

        if self.augmentation:
            augmented = self.augmentation(image=image, mask=heatmap)
            image = augmented['image']
            heatmap = augmented['mask']

        image = torch.from_numpy(image).permute(2, 0, 1) # (3,800,640)
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1) # (20,800,640)

        # Повертаємо 3 копії heatmap під multi-output модель.
        return image, (heatmap, heatmap, heatmap)