# Quantum-Chip

This repository offers a minimal PyTorch pipeline for estimating the performance of superconducting quantum chips from paired TEM and AFM images.

## Contents

1. [Dataset](#dataset)
2. [Training](#training)
3. [Interactive options](#interactive-options)
4. [Speech notifications](#speech-notifications)

## Dataset

Create a CSV file where each row describes a chip:

```
tem_path,afm_path,performance
/path/to/chip1_tem.png,/path/to/chip1_afm.png,0.85
```

Each image pair is loaded, resized to 224×224, stacked as a two-channel tensor and mapped to the numeric performance value.

## Training

```
python train.py --csv path/to/data.csv --epochs 20 --output model.pt
```

The script reads the CSV, trains a modified ResNet‑18 and saves weights to `model.pt`.

### Interactive options

- `--ui` opens a file dialog to choose the CSV.
- `--plot` shows a live plot of training/validation loss and progress bars with ETA.

## Speech notifications

Add spoken feedback with language buttons:

```
python train.py --csv path/to/data.csv --speak
```

Click the desired button in the pop-up window to switch among:

| Code | Language |
|------|----------|
| zh   | 中文 |
| en   | English |
| de   | Deutsch |
| fr   | Français |
| es   | Español |
| el   | Ελληνικά |

Every epoch completion is announced in the selected language (requires `pyttsx3`).
