import os, copy, time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

data_dir = "data"  # <-- change if needed
num_epochs = 15
batch_size = 64
img_size = 224
lr = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Augmentation / normalization (ImageNet stats)
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

val_tfms = transforms.Compose([
    transforms.Resize(int(img_size*1.15)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=val_tfms)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
class_names = train_ds.classes
num_classes = len(class_names)

# Load pretrained MobileNetV3-Small
weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = models.mobilenet_v3_small(weights=weights)

# Replace classifier for our classes
in_feats = model.classifier[3].in_features  # last Linear
model.classifier[3] = nn.Linear(in_feats, num_classes)

# Option A: fine-tune all layers (best if you have enough images)
# Option B: freeze backbone for first epochs, then unfreeze:
# for p in model.features.parameters():
#     p.requires_grad = False

model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

best_acc, best_wts = 0.0, copy.deepcopy(model.state_dict())

def run_epoch(loader, train=True):
    model.train(train)
    epoch_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for images, labels in loader:
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
          f"train {tr_loss:.4f}/{tr_acc:.3f}  val {val_loss:.4f}/{val_acc:.3f}  "
          f"time {time.time()-t0:.1f}s")

print(f"Best val acc: {best_acc:.3f}")
model.load_state_dict(best_wts)
torch.save({"state_dict": model.state_dict(),
            "classes": class_names}, "mobilenetv3_small_people.pt")
