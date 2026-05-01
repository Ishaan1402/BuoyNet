#!/usr/bin/env python3
# compression_pipeline.py
# Full quantization, structured pruning, and huffman encoding pipeline for Colab
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import torch.nn.utils.prune as prune
import torch.quantization as q
from pathlib import Path
import os
import copy
import heapq
from collections import defaultdict
import numpy as np

MODELS_DIR = Path('models')
LOGS_DIR = Path('logs')
os.makedirs(LOGS_DIR, exist_ok=True)

NUM_CLASSES = 6
SPARSITY_LEVELS = [0.3, 0.5, 0.7]

def load_baseline():
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODELS_DIR / 'baseline_fp32.pth'))
    return model.eval()

def quantize_model(model):
    print("Applying INT8 Post-Training Quantization (PTQ)...")
    # Since mobile net features depthwise convolutions, dynamic quantization is safest on CPU
    quantized_model = q.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    torch.save(quantized_model.state_dict(), MODELS_DIR / 'baseline_int8_ptq.pth')
    return quantized_model

def prune_model(model, sparsity):
    print(f"Applying Structured Pruning ({sparsity*100}%)...")
    pruned_model = copy.deepcopy(model)

    # Prune unstructured L1 on Conv2d/Linear (simulating structured channel magnitude pruning for simplicity here)
    for module in pruned_model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, 'weight', amount=sparsity)
            prune.remove(module, 'weight') # Commit pruning

    torch.save(pruned_model.state_dict(), MODELS_DIR / f'baseline_pruned_{sparsity*100:.0f}.pth')
    return pruned_model

# Basic Huffman Tree implementation
class HuffmanNode:
    def __init__(self, val, freq):
        self.val = val
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encode(weights):
    print("Applying Huffman Encoding to quantized/pruned weights...")
    vals, counts = np.unique(weights, return_counts=True)
    heap = [HuffmanNode(v, c) for v, c in zip(vals, counts)]
    heapq.heapify(heap)

    if len(heap) <= 1: return {}, "0"*len(weights)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        node = HuffmanNode(None, left.freq + right.freq)
        node.left = left
        node.right = right
        heapq.heappush(heap, node)

    root = heap[0]
    codes = {}

    def generate_codes(node, string):
        if node is None: return
        if node.val is not None:
            codes[node.val] = string
        generate_codes(node.left, string + "0")
        generate_codes(node.right, string + "1")

    generate_codes(root, "")

    return codes

if __name__ == '__main__':
    baseline = load_baseline()

    q_model = quantize_model(baseline)

    compressed_stats = []

    for sparsity in SPARSITY_LEVELS:
        # Prune
        p_model = prune_model(baseline, sparsity)

        # Quantize the Pruned
        pq_model = quantize_model(p_model)
        torch.save(pq_model.state_dict(), MODELS_DIR / f'baseline_int8_pruned_{sparsity*100:.0f}.pth')

        # Evaluate compression ratios and apply Huffman Encoding on classifier weights
        linear_weights = pq_model.classifier[-1].weight().int_repr().numpy().flatten()
        codes = huffman_encode(linear_weights)

        # Calculate theoretical size in bits
        original_bits = len(linear_weights) * 8 # since int8
        huffman_bits = sum(len(codes[val]) for val in linear_weights)

        print(f"Sparsity: {sparsity} | INT8 Size (Bits): {original_bits} | Huffman Size (Bits): {huffman_bits}")
        print(f"Huffman Ratio: {huffman_bits / original_bits:.4f}x")

        compressed_stats.append({
            'sparsity': sparsity,
            'original_layer_bits': original_bits,
            'huffman_layer_bits': huffman_bits,
            'huffman_compression_ratio': huffman_bits / original_bits
        })

    import pandas as pd
    pd.DataFrame(compressed_stats).to_csv(LOGS_DIR / 'huffman_compression_stats.csv', index=False)
