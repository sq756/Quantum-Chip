import csv
from pathlib import Path
from typing import Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SuperconductingChipDataset(Dataset):
    """Dataset for TEM/AFM images and chip performance labels."""

    def __init__(self, csv_file: str, transform: Optional[transforms.Compose] = None) -> None:
        self.csv_path = Path(csv_file)
        self.transform = transform or transforms.ToTensor()
        self.samples = []
        with self.csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Expect columns: tem_path, afm_path, performance
                self.samples.append(
                    {
                        "tem_path": Path(row["tem_path"]),
                        "afm_path": Path(row["afm_path"]),
                        "performance": float(row["performance"]),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        tem_img = Image.open(sample["tem_path"]).convert("L")
        afm_img = Image.open(sample["afm_path"]).convert("L")
        tem_tensor = self.transform(tem_img)
        afm_tensor = self.transform(afm_img)
        image = torch.cat([tem_tensor, afm_tensor], dim=0)  # (2, H, W)
        performance = torch.tensor(sample["performance"], dtype=torch.float32)
        return image, performance
