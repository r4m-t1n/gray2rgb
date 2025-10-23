import os
from PIL import Image

import torch
from torchvision import transforms

IMG_SIZE = 64

class IMGDataset(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))])

        gray_transforms_list = [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Grayscale(num_output_channels=1)
        ]

        if folder.endswith("train"):
            gray_transforms_list += [
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1)
            ]

        gray_transforms_list += [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]

        self.gray_transform = transforms.Compose(gray_transforms_list)

        self.color_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        gray = self.gray_transform(img)
        original = self.color_transform(img)
        return gray, original