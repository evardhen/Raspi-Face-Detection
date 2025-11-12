import os, copy, time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils import make_tfms, get_classes
from utils.model_loader import load_model

data_dir = "data/images"  # Each person has own subfolder
num_epochs = 15
batch_size = 4
lr = 3e-4
plot_example = False
model_type = "mobilefacenet"  # "mobilefacenet" or "arcface_r50"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    num_classes = len(get_classes())
    model, img_size = load_model(model_type, num_classes)
    model.to(device)

    train_tfms = make_tfms(model_type=model_type, train=True, img_size=img_size)
    val_tfms   = make_tfms(model_type=model_type, train=False, img_size=img_size)

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=val_tfms)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)
    class_names = train_ds.classes
    print(f"Found {len(train_ds)} training images in {len(class_names)} classes")
    # quick sanity check
    print("model on:", next(model.parameters()).device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc, best_wts = 0.0, copy.deepcopy(model.state_dict())

    def run_epoch(loader, train=True):
        model.train(train)
        epoch_loss, correct, total = 0.0, 0, 0
        torch.set_grad_enabled(train)
        for b, (images, labels) in enumerate(loader):

            if train and images.size(0) == 1:
                print(f"[WARN] skipping batch {b}: batch_size=1 (BN unsafe)")
                continue
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return epoch_loss/total, correct/total

    for epoch in range(num_epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader,   train=False)
        scheduler.step()
        if val_acc > best_acc:
            best_acc, best_wts = val_acc, copy.deepcopy(model.state_dict())
        print(f"Epoch {epoch+1:02d}/{num_epochs} "
            f"train loss: {tr_loss:.4f} train acc: {tr_acc:.3f}  val loss: {val_loss:.4f} val acc: {val_acc:.3f}  "
            f"time {time.time()-t0:.1f}s")

    print(f"Best val acc: {best_acc:.3f}")
    model.load_state_dict(best_wts)
    if model_type == "mobilefacenet":
        torch.save({"state_dict": model.state_dict(),
                    "classes": class_names}, "saves/mobilefacenet_people.pt")
    elif model_type == "arcface_r50":
        torch.save({"state_dict": model.state_dict(),
                    "classes": class_names}, "saves/arcface_r50_people.pt")
    elif model_type == "mobilenetv3_small":
        torch.save({"state_dict": model.state_dict(),
                    "classes": class_names}, "saves/mobilenetv3_small_people.pt")
    


if __name__ == "__main__":
    print(f"Using device: {device}")
    print("cuda in torch wheel:", torch.version.cuda)
    print(f"Model type: {model_type}")
    torch.multiprocessing.set_start_method("spawn", force=True)  # <— key line
    main()
