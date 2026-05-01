#!/usr/bin/env python3
# domain_shift.py
# Simulates turbidity, biofouling, and lighting variations on test datastore
import cv2
import numpy as np
import os
from pathlib import Path
import random

DATA_DIR = Path('data')
TEST_DIR = DATA_DIR / 'test'
AUG_DIR = DATA_DIR / 'test_domain_shift'

def simulate_turbidity(image, turbidity_level=0.3):
    h, w = image.shape[:2]
    kernel_size = int(5 + turbidity_level * 10)
    # Ensure kernel size is odd
    if kernel_size % 2 == 0: kernel_size += 1

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    alpha = 1 - turbidity_level
    blended = cv2.addWeighted(image, alpha, blurred, turbidity_level, 0)

    # Introduce suspended particles simulation (noise)
    noise = np.random.normal(0, turbidity_level * 50, image.shape)
    blended = np.clip(blended + noise, 0, 255).astype(np.uint8)
    return blended

def simulate_biofouling(image, biofouling_level=0.2):
    """
    Biological film/algae coating simulation on lens by applying green/brown translucency
    """
    overlay = np.zeros_like(image)
    overlay[:, :, 1] = 100  # Green
    overlay[:, :, 0] = 50   # Blue

    alpha = biofouling_level
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return blended

def simulate_lighting_variation(image, lighting_factor=0.7):
    """
    Simulate poor/variable underwater lighting with Alpha + Beta shifts
    """
    return cv2.convertScaleAbs(image, alpha=lighting_factor, beta=10)

def process_test_dataset():
    if not TEST_DIR.exists():
        print("No test dataset found to augment! Please run `prepare_dataset.py` to acquire test images.")
        return

    os.makedirs(AUG_DIR, exist_ok=True)
    images = list(TEST_DIR.rglob('*.jpg'))

    transforms = [
        ('turbid_high', lambda x: simulate_turbidity(x, 0.7)),
        ('biofoul_low', lambda x: simulate_biofouling(x, 0.2)),
        ('light_dim', lambda x: simulate_lighting_variation(x, 0.5)),
    ]

    print(f"Generating domain shift variations for {len(images)} images. Configurations: {len(transforms)}")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None: continue

        rel_path = img_path.relative_to(TEST_DIR)

        for name, transform in transforms:
            aug_img = transform(img)

            save_path = AUG_DIR / name / rel_path
            os.makedirs(save_path.parent, exist_ok=True)
            cv2.imwrite(str(save_path), aug_img)

if __name__ == '__main__':
    process_test_dataset()
