import argparse
import threading
import time

import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm

from dataset import SuperconductingChipDataset
from model import ChipPerformanceNet
from speech import LANGUAGES, speak_message


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def train(args, selected_lang):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = SuperconductingChipDataset(args.csv, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = get_device()
    model = ChipPerformanceNet(pretrained=True).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_losses, epoch_times = [], [], []

    if args.plot:
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots()

    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False)
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())
        running_loss /= len(train_loader.dataset)
        train_losses.append(running_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                preds = model(images)
                val_loss += criterion(preds, targets).item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        epoch_time = time.time() - start
        epoch_times.append(epoch_time)
        remaining = (args.epochs - epoch - 1) * (sum(epoch_times) / len(epoch_times))
        print(f"Epoch {epoch + 1}: val_loss={val_loss:.4f} ETA:{remaining:.1f}s")
        if args.speak:
            speak_message(selected_lang["code"], "epoch", epoch=epoch + 1, total=args.epochs, val_loss=val_loss)

        if args.plot:
            ax.clear()
            ax.plot(range(1, len(train_losses) + 1), train_losses, label="train")
            ax.plot(range(1, len(val_losses) + 1), val_losses, label="val")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            plt.pause(0.001)

    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Train chip performance predictor")
    parser.add_argument("--csv", type=str, help="CSV with tem_path, afm_path, performance")
    parser.add_argument("--output", type=str, default="model.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ui", action="store_true", help="Use file dialog to select CSV if not provided")
    parser.add_argument("--plot", action="store_true", help="Show real-time training plot")
    parser.add_argument("--speak", action="store_true", help="Enable voice notifications with language buttons")
    args = parser.parse_args()

    if args.ui or not args.csv:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file_path:
                args.csv = file_path
        except Exception as exc:
            raise RuntimeError(f"File dialog failed: {exc}")
    if not args.csv:
        parser.error("CSV file must be provided")

    selected_lang = {"code": "en"}
    if args.speak:
        try:
            import tkinter as tk
            root = tk.Tk()
            root.title("Select language")
            for code, data in LANGUAGES.items():
                tk.Button(root, text=data["name"], command=lambda c=code: selected_lang.__setitem__("code", c)).pack(side=tk.LEFT)
            threading.Thread(target=root.mainloop, daemon=True).start()
        except Exception as exc:
            print(f"Language selector failed: {exc}")
    train(args, selected_lang)


if __name__ == "__main__":
    main()
