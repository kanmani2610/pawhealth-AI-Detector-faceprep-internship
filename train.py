
import os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

TRAIN_DIR  = "dataset/train"
VALID_DIR  = "dataset/valid"
MODEL_PATH = "model/dog_health_model.pth"
IMG_SIZE   = 224
BATCH_SIZE = 16
NUM_EPOCHS = 15
LR         = 1e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_ds   = datasets.ImageFolder(VALID_DIR, transform=val_tf)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

CLASS_NAMES  = train_ds.classes
NUM_CLASSES  = len(CLASS_NAMES)
print(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")
print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}  |  Device: {DEVICE}")

model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

best_val_acc = 0.0
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    rl = rc = rt = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward(); optimizer.step()
        rl += loss.item() * imgs.size(0)
        _, p = out.max(1)
        rc += p.eq(labels).sum().item(); rt += imgs.size(0)

    model.eval(); vc = vt = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            _, p = model(imgs).max(1)
            vc += p.eq(labels).sum().item(); vt += imgs.size(0)

    val_acc = vc / vt; scheduler.step()
    print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  loss:{rl/rt:.4f}  train:{rc/rt:.3f}  val:{val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("model", exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "classes": CLASS_NAMES}, MODEL_PATH)
        print(f"  ✓ Saved (val_acc={val_acc:.3f})")

print(f"\nDone. Best val acc: {best_val_acc:.3f}  →  {MODEL_PATH}")