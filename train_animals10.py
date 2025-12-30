import os
from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

@dataclass
class Config:
    data_dir: str = "data/animals10"
    epochs: int = 8
    batch_size: int = 32
    lr: float = 3e-4
    num_workers: int = 2
    model_path: str = "models/animals10_resnet18.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def get_loaders(cfg: Config):
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop((224, 224), scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])

    train_ds = datasets.ImageFolder(os.path.join(cfg.data_dir, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(cfg.data_dir, "val"), transform=eval_tf)
    test_ds  = datasets.ImageFolder(os.path.join(cfg.data_dir, "test"), transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    return train_loader, val_loader, test_loader, train_ds.classes

def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().tolist()
        y_pred.extend(preds)
        y_true.extend(y.tolist())

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return acc, precision, recall, f1

def train():
    cfg = Config()
    os.makedirs(os.path.dirname(cfg.model_path), exist_ok=True)

    train_loader, val_loader, test_loader, classes = get_loaders(cfg)
    model = build_model(num_classes=len(classes)).to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        for x, y in pbar:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.item()))

        val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, cfg.device)
        print(f"[VAL] acc={val_acc:.4f} precision={val_prec:.4f} recall={val_rec:.4f} f1={val_f1:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model_state": model.state_dict(), "classes": classes}, cfg.model_path)
            print(f"✅ Best model saved: {cfg.model_path}")

    # Test
    ckpt = torch.load(cfg.model_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state"])
    test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, cfg.device)
    print(f"[TEST] acc={test_acc:.4f} precision={test_prec:.4f} recall={test_rec:.4f} f1={test_f1:.4f}")
    print("Sınıflar:", classes)

if __name__ == "__main__":
    train()
