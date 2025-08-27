# Quantum-Chip

This repository provides a minimal PyTorch pipeline for estimating the performance of superconducting quantum chips using paired TEM and AFM images.

## Dataset format

Create a CSV file where each row describes a chip with the following columns:

```
tem_path,afm_path,performance
/path/to/chip1_tem.png,/path/to/chip1_afm.png,0.85
```

## Training

```
python train.py --csv path/to/data.csv --epochs 20 --output model.pt
```

The script reads the CSV, stacks each chip's TEM and AFM images as a two-channel tensor, and trains a modified ResNet-18 to predict the measured performance value.
