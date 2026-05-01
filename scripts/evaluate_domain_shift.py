#!/usr/bin/env python3
# evaluate_domain_shift.py
# Evaluates baseline and compressed models against clean and domain-shifted test datasets.
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import torch.quantization as q
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pandas as pd
import os
from pathlib import Path
from PIL import Image, UnidentifiedImageError

MODELS_DIR = Path('models')
RESULTS_DIR = Path('figures')
DATA_DIR = Path('data')
TEST_DIR = DATA_DIR / 'test'
AUG_DIR = DATA_DIR / 'test_domain_shift'

NUM_CLASSES = 16
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Testing Environment evaluated on Device: {device}")

def safe_pil_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except Exception as e:
        return Image.new('RGB', (224, 224), (0, 0, 0))

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_baseline():
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    return model

def evaluate_model(model_name, model, dataloader, is_quantized=False):
    model.eval()
    if not is_quantized:
        model = model.to(device)
    else:
        # Quantized models must run on CPU in standard pytorch runtime
        model = model.to('cpu')

    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            if not is_quantized:
                imgs, labels = imgs.to(device), labels.to(device)
            else:
                imgs, labels = imgs.to('cpu'), labels.to('cpu')

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).float().sum().item()
            total += labels.size(0)

    if total == 0: return 0.0
    return correct / total * 100.0

def generate_environment_dataloaders():
    loaders = {}

    # 1. Clean Test
    if TEST_DIR.exists():
        ds = ImageFolder(root=str(TEST_DIR), transform=val_transform, loader=safe_pil_loader)
        loaders['clean'] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Augmented Domains
    if AUG_DIR.exists():
        for dom_path in AUG_DIR.iterdir():
            if dom_path.is_dir():
                ds = ImageFolder(root=str(dom_path), transform=val_transform, loader=safe_pil_loader)
                loaders[dom_path.name] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    return loaders

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    loaders = generate_environment_dataloaders()

    if not loaders:
        print("No test datasets found! Run split/augmentation pipelines first.")
        exit(1)

    print(f"Loaded {len(loaders)} testing environments.")

    model_paths = list(MODELS_DIR.glob('*.pth'))
    results = []

    for mpath in model_paths:
        mname = mpath.stem
        print(f"\n--- Loading {mname} ---")

        is_quant = 'int8' in mname
        model = load_baseline()

        if is_quant:
            model = q.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

        try:
            model.load_state_dict(torch.load(mpath, map_location='cpu'))
        except Exception as e:
            print(f"Failed to load {mname}. Skipping. {str(e)}")
            continue

        for env_name, dl in loaders.items():
            print(f"  Evaluating on {env_name}...")
            acc = evaluate_model(mname, model, dl, is_quantized=is_quant)
            print(f"    Accuracy: {acc:.2f}%")

            results.append({
                'model_name': mname,
                'test_condition': env_name,
                'accuracy': acc
            })

    # Serialize results
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / 'accuracy_metrics.csv', index=False)
    print(f"\nAccuracy evaluation completed. Exported to {RESULTS_DIR / 'accuracy_metrics.csv'}")
