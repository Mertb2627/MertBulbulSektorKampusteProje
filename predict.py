import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

def load_model(model_path: str, device: str):
    ckpt = torch.load(model_path, map_location=device)
    classes = ckpt["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, classes

def preprocess_pil(img: Image.Image):
    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        # ✅ ImageNet normalize (Animals-10 train_animals10.py ile aynı)
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])
    return tf(img.convert("RGB")).unsqueeze(0)
