import csv
import sys
from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms as transforms

# Ensure project root on path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from dataset import SuperconductingChipDataset
from model import ChipPerformanceNet


def create_dummy_data(tmpdir: Path) -> Path:
    tem_path = tmpdir / "tem.png"
    afm_path = tmpdir / "afm.png"
    Image.new("L", (32, 32), color=0).save(tem_path)
    Image.new("L", (32, 32), color=0).save(afm_path)

    csv_path = tmpdir / "data.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tem_path", "afm_path", "performance"])
        writer.writeheader()
        writer.writerow({"tem_path": str(tem_path), "afm_path": str(afm_path), "performance": "0.9"})
    return csv_path


def test_dataset_and_model(tmp_path):
    csv_file = create_dummy_data(tmp_path)
    ds = SuperconductingChipDataset(csv_file, transform=transforms.ToTensor())
    image, target = ds[0]
    assert image.shape[0] == 2
    model = ChipPerformanceNet(pretrained=False)
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))
    assert output.shape == torch.Size([1])
