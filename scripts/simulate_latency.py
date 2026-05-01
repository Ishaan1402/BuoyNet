#!/usr/bin/env python3
# simulate_latency.py
# Evaluate and Benchmark latency/energy on simulated Pi hardware running from Colab
import time
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import torch.quantization as q
import pandas as pd
from pathlib import Path
import os
from typing import Dict

# Configuration
MODELS_DIR = Path("models")
RESULTS_DIR = Path("figures")
OS_RESERVE_RAM_MB = 500  # 4GB Pi approximation
PI_LATENCY_SCALAR = 3.5  # Estimate from typical Colab K80/T4 vs Pi CPU throughput
NUM_CLASSES = 16
NUM_RUNS = 100

def generate_pi_scaling():
    return PI_LATENCY_SCALAR

def get_base_model():
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    return model

def profile_model(model_name: str, model: nn.Module) -> Dict:
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # Warm-up
    print(f"Warming up profile runs for {model_name}...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    print("Measuring raw inference latency (Colab)...")
    with torch.no_grad():
        for _ in range(NUM_RUNS):
            start = time.perf_counter()
            _ = model(dummy_input)
            if torch.cuda.is_available() and device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    # Estimate Pi metrics
    colab_mean_ms = sum(times) / NUM_RUNS
    pi_inference_ms = colab_mean_ms * generate_pi_scaling()

    # Simulating standard Edge IO costs
    camera_capture_ms = 200 # 5 FPS realistic budget
    preprocessing_ms = 75   # CV2 Resize/Normalize
    postprocessing_ms = 50  # Results packaging + Network I/O
    e2e_pi_latency_ms = pi_inference_ms + camera_capture_ms + preprocessing_ms + postprocessing_ms

    # Storage and Power analysis
    # Measure exact non-zero parameters for actual footprint scaling
    param_count = sum((p != 0).sum().item() for p in model.parameters())
    fp32_size_mb = (param_count * 4) / (1024 ** 2)

    # Manual Adjustment for Quantized Bit-width
    if 'int8' in model_name:
        fp32_size_mb = fp32_size_mb * 0.25 # ints are 8 bits, floats are 32

    # Approximated power profile of Pi (Idle 2W, Load 5W peak)
    load_power_w = 4.5
    energy_inference_joules = load_power_w * (pi_inference_ms / 1000)

    return {
        "model_name": model_name,
        "colab_base_inference_ms": colab_mean_ms,
        "pi_estimated_inference_ms": pi_inference_ms,
        "pi_e2e_latency_ms": e2e_pi_latency_ms,
        "pi_max_fps": 1000 / e2e_pi_latency_ms,
        "estimated_size_mb": fp32_size_mb,
        "energy_consumption_joules": energy_inference_joules
    }

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []

    for m_path in MODELS_DIR.glob('*.pth'):
        m_name = m_path.stem
        is_quant = 'int8' in m_name

        model = get_base_model()

        if is_quant:
            model = q.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

        try:
            model.load_state_dict(torch.load(m_path, map_location='cpu' if is_quant else None))
            if not is_quant and torch.cuda.is_available():
                model = model.to('cuda')
        except Exception as e:
            print(f"Failed loading {m_name}, skipping.")
            continue

        res = profile_model(m_name, model)
        results.append(res)

    pd.DataFrame(results).to_csv(RESULTS_DIR / 'pi_latency_estimates.csv', index=False)
    print("Latency and memory estimation complete. Results mapped to file.")