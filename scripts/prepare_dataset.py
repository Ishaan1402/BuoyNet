#!/usr/bin/env python3
# prepare_dataset.py
import os
import urllib.request
import zipfile
import shutil
import numpy as np
from pathlib import Path
import json
import cv2

DATA_DIR = Path('data')
RAW_DIR = DATA_DIR / 'raw'
SPLITS_DIR = DATA_DIR / 'splits'

def setup_directories():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(DATA_DIR / split, exist_ok=True)

def reorganize_frontiers_hierarchy():
    """
    The Frontiers dataset unzips into MACRO/MESO/MICRO folders containing sub-class folders.
    PyTorch ImageFolder needs a flat `dataset/class_name/images.jpg` structure.
    This function flattens the hierarchy and aggregates the individual particle images.
    """
    print("Flattening Frontiers dataset hierarchy for PyTorch...")
    classes_dir = DATA_DIR / 'classes'
    os.makedirs(classes_dir, exist_ok=True)

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

    # 1. Identify all images inside the size boundaries
    paths = [p for p in RAW_DIR.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS]

    # Filters
    valid_paths = [p for p in paths if os.path.getsize(p) > 0]

    image_count = 0

    for path in valid_paths:
        #  structure is dataset/SIZE/class_name/img.jpg
        # Example: dataset/MICRO/pellet/img.jpg OR dataset/MACRO/bottle_ma/img.jpg
        parts = path.parts

        if 'raw_img' in parts or 'annotation' in parts or 'reference' in parts or 'reference_ma' in parts or 'reference_me' in parts:
            continue

        # Find the parent folder which is usually the class name
        class_name = path.parent.name.upper()

        # Some classes have size suffixes like 'cap_me' or 'bottle_ma'. Strip them to merge MACRO/MESO classes
        if class_name.endswith('_ME') or class_name.endswith('_MA'):
            class_name = class_name[:-3]

        dest_dir = classes_dir / class_name
        os.makedirs(dest_dir, exist_ok=True)

        # Move file safely
        dest_path = dest_dir / f"{path.parent.parent.name}_{path.name}" # prepend size prefix to prevent naming collisions
        shutil.copy2(path, dest_path)
        image_count += 1

    print(f"Successfully reorganized {image_count} labeled particle crops into {len(list(classes_dir.iterdir()))} unified PyTorch classes.")
    return classes_dir

def mock_dataset_if_empty():
    classes = ['PET', 'HDPE', 'LDPE', 'PP', 'PS', 'PVC']
    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    valid_found = [p for p in RAW_DIR.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS]

    if not valid_found:
        print("Creating placeholder dataset structure...")
        for c in classes:
            os.makedirs(RAW_DIR / c, exist_ok=True)
            # Create empty dummy files
            bg = np.zeros((224, 224, 3), dtype=np.uint8)
            for i in range(10):
                path = str(RAW_DIR / c / f'{c}_{i}.jpg')
                import cv2
                cv2.imwrite(path, bg)
    else:
        print(f"Real dataset found in raw directory! ({len(valid_found)} images). Skipping mock generation.")

def create_splits(source_dir):
    # Simple random split
    indices = {"train": [], "val": [], "test": []}

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    paths = [p for p in source_dir.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS]

    if len(paths) == 0:
        print("No valid images found to split!")
        return

    np.random.shuffle(paths)
    n = len(paths)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    try:
        indices["train"] = [str(p.relative_to(source_dir)) for p in paths[:train_end]]
        indices["val"] = [str(p.relative_to(source_dir)) for p in paths[train_end:val_end]]
        indices["test"] = [str(p.relative_to(source_dir)) for p in paths[val_end:]]
    except Exception as e:
        print("Relative path evaluation failed. Pathing error.", e)
        return

    with open(SPLITS_DIR / 'indices.json', 'w') as f:
        json.dump(indices, f, indent=4)

    print(f"Created splits: {len(indices['train'])} train, {len(indices['val'])} val, {len(indices['test'])} test")

    # Copy files from 'classes' dir to the final Train/Val/Test PyTorch staging area
    for split_key, split_paths in indices.items():
        split_dir = DATA_DIR / split_key
        for rel_path in split_paths:
            src = source_dir / rel_path
            dst = split_dir / rel_path
            os.makedirs(dst.parent, exist_ok=True)
            shutil.copy2(src, dst)

if __name__ == '__main__':
    setup_directories()
    mock_dataset_if_empty()
    classes_unified_dir = reorganize_frontiers_hierarchy()
    create_splits(classes_unified_dir)
