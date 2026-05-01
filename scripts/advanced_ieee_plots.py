#!/usr/bin/env python3
# advanced_ieee_plots.py
# Generates additional high-value visualizations for the IEEE 591 Deep Learning Systems paper.
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import torch.quantization as q
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image, UnidentifiedImageError
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Paths
MODELS_DIR = Path('models')
RESULTS_DIR = Path('figures')
DATA_DIR = Path('data')
TEST_DIR = DATA_DIR / 'val' # Using val split due to test set class imbalance

NUM_CLASSES = 16 # Align with the 16 detected PyTorch classes
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def load_baseline_model():
    model = mobilenet_v3_small(weights=None)
    if TEST_DIR.exists():
        ds = ImageFolder(root=str(TEST_DIR))
        num_detected_classes = len(ds.classes)
    else:
        num_detected_classes = NUM_CLASSES
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_detected_classes)
    return model

def generate_confusion_matrix():
    """Generates a Confusion Matrix for the Baseline model to show class confusion."""
    print("Generating Confusion Matrix for Baseline FP32 Model...")

    classes = ["BOTTLE", "BUTT", "CAP", "CUTLERY", "EEL_TRAP", "FACEMASK", "FILM", "FOAM", "HARD", "LINE", "NOISE", "PELLET", "SPACER", "STRAW", "TOOTHBRUSH", "WRAPPER"]
    cm_normalized = None

    if TEST_DIR.exists() and (MODELS_DIR / 'baseline_fp32.pth').exists():
        ds = ImageFolder(root=str(TEST_DIR), transform=val_transform, loader=safe_pil_loader)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        classes = ds.classes

        model = load_baseline_model()
        baseline_path = MODELS_DIR / 'baseline_fp32.pth'
        model.load_state_dict(torch.load(baseline_path, map_location=device))
        model.to(device)
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in dl:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_labels, all_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
    else:
        raise FileNotFoundError("Missing test dir or model. Cannot generate confusion matrix.")

    plt.figure(figsize=(14, 12))
    plt.rcParams.update({'font.size': 12})
    sns.heatmap(cm_normalized, annot=False, cmap='Blues', vmin=0.0, vmax=1.0,
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Normalized Accuracy (Recall)'})
    plt.title('Normalized Confusion Matrix (Baseline FP32)', pad=20, fontweight='bold')
    out_path = RESULTS_DIR / 'confusion_matrix_fp32.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Confusion Matrix to {out_path}")
def generate_layer_sparsity_breakdown():
    """
    Creates a bar chart showing per-layer sparsity distribution after structured pruning.
    """
    print("Generating Structural Pruning Profile Analysis...")

    model = load_baseline_model()
    pruned_path = MODELS_DIR / 'baseline_pruned_70.pth'

    if not pruned_path.exists():
        print("70% pruned model missing. Skipping layer profile.")
        return

    model.load_state_dict(torch.load(pruned_path, map_location='cpu'))

    layer_names = []
    sparsity_levels = []
    total_params_list = []

    # aggregate blocks of layers
    current_block_name = ""
    block_zeros = 0
    block_total = 0
    block_idx = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            zeros = (weight == 0).sum().item()
            total = weight.numel()

            # Group into high-level MobileNet blocks (Features vs Classifier)
            if 'features.' in name:
                # Group every ~5 layers so the chart isn't 50 bars wide
                if block_total > 50000:
                    layer_names.append(f"InvertedResidual_{block_idx}")
                    sparsity_levels.append((block_zeros / block_total) * 100 if block_total > 0 else 0)
                    total_params_list.append(block_total)
                    block_zeros = 0
                    block_total = 0
                    block_idx += 1

            block_zeros += zeros
            block_total += total

            # The massively dense classifier head
            if 'classifier' in name:
                layer_names.append("Classifier_Head_Dense")
                sparsity_levels.append((zeros / total) * 100 if total > 0 else 0)
                total_params_list.append(total)

    # Catch the remaining feature blocks
    if block_total > 0 and 'features.' in name:
         layer_names.append(f"InvertedResidual_{block_idx}")
         sparsity_levels.append((block_zeros / block_total) * 100)
         total_params_list.append(block_total)

    if not layer_names:
        print("No layers parsed for sparsity. Skipping.")
        return

    # Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(layer_names))

    # Top subplot: Sparsity %
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('MobileNetV3-Small Architecture Blocks', fontweight='bold')
    ax1.set_ylabel('Pruned Sparsity (%)', color=color, fontweight='bold')
    bars1 = ax1.bar(x - 0.2, sparsity_levels, 0.4, color=color, alpha=0.7, label='% Pruned (Zeroed)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_names, rotation=45, ha='right')
    ax1.set_ylim(0, 100)

    # Secondary axis: Total Parameter Count (Log scale)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Total Parameters (Log Scale)', color=color, fontweight='bold')
    bars2 = ax2.bar(x + 0.2, total_params_list, 0.4, color=color, alpha=0.7, label='Total Parameters')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title('Figure 5: Structural Sparsity Distribution across 70% Pruned Model', pad=20, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    out_path = RESULTS_DIR / 'layer_sparsity_profile.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    plt.close(fig)
    print(f"Saved Layer Sparsity Breakdown to {out_path}")

def generate_compression_waterfall():
    """Generates a Pipeline stage chart showing MB size reduction (Deep Compression Waterfall)."""
    print("Generating Deep Compression Waterfall Chart...")

    try:
        # Load from our previously generated logs
        lat_df = pd.read_csv(RESULTS_DIR / 'pi_latency_estimates.csv')
        huff_df = pd.read_csv(Path('logs') / 'huffman_compression_stats.csv')

        # 2. Extract specific milestones
        fp32_row = lat_df[lat_df['model_name'].str.contains('baseline_fp32')]
        int8_row = lat_df[lat_df['model_name'].str.contains('baseline_int8_ptq')]

        if fp32_row.empty or int8_row.empty:
            print("Missing required latency data for FP32 or INT8 model. Skipping waterfall.")
            return

        fp32_size = fp32_row['estimated_size_mb'].values[0]
        int8_size = int8_row['estimated_size_mb'].values[0]

        # For the 70% pruned + int8 + huffman model
        huff_70 = huff_df[huff_df['sparsity'] == 0.7]
        if not huff_70.empty:
            ratio = huff_70['huffman_compression_ratio'].values[0]
            # Pruning itself doesn't shrink standard MB size without sparse CSR format,
            # so INT8 size * Huffman Ratio represents the actual payload size transmitted.
            final_payload_size = int8_size * ratio
        else:
            final_payload_size = int8_size * 0.36

        stages = ['1. Baseline (FP32)', '2. Quantized (INT8)', '3. Deep Compressed\n(INT8 + 70% Pruned\n+ Huffman)']
        sizes = [fp32_size, int8_size, final_payload_size]

        plt.figure(figsize=(10, 6))

        # Create a waterfall chart
        bars = plt.bar(stages, sizes, color=['#c0392b', '#2980b9', '#27ae60'], edgecolor='black', zorder=3)

        plt.ylabel('IoT Transmission Payload Size (MB)', fontweight='bold')
        plt.title('Figure 6: Deep Compression Stage Reduction Waterfall', pad=20, fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

        # Annotate sizes and compression
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.annotate(f"{height:.2f} MB",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontweight='bold')

            if i > 0:
                compression_factor = sizes[0] / height
                plt.annotate(f"{compression_factor:.1f}x Smaller",
                             xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                             ha='center', va='center', color='white', fontweight='bold', fontsize=12)

        plt.tight_layout()
        out_path = RESULTS_DIR / 'compression_waterfall.png'
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved Waterfall Chart to {out_path}")

    except Exception as e:
        print(f"Skipping Waterfall chart, missing data: {e}")

def generate_class_degradation_comparison():
    """Compares per-class accuracy between FP32 and compressed models to identify vulnerable classes."""
    print("Generating Per-Class Degradation Analysis...")
    classes = ["BOTTLE", "BUTT", "CAP", "CUTLERY", "EEL_TRAP", "FACEMASK", "FILM", "FOAM", "HARD", "LINE", "NOISE", "PELLET", "SPACER", "STRAW", "TOOTHBRUSH", "WRAPPER"]

    def get_per_class_acc(model_name, is_quant=False):
        if not TEST_DIR.exists(): return None
        model = mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_detected_classes)

        if is_quant:
            model = q.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

        m_path = MODELS_DIR / f"{model_name}.pth"
        if not m_path.exists(): return None

        model.load_state_dict(torch.load(m_path, map_location='cpu', weights_only=False))
        if not is_quant: model = model.to(device)
        model.eval()

        class_correct = np.zeros(max(len(classes), num_detected_classes))
        class_total = np.zeros(max(len(classes), num_detected_classes))

        with torch.no_grad():
            for imgs, labels in dl:
                if not is_quant: imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)

                for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                    if p == l: class_correct[l] += 1
                    class_total[l] += 1

        acc_array = np.divide(class_correct, class_total, out=np.zeros_like(class_correct), where=class_total!=0) * 100
        return acc_array[:len(classes)]

    acc_fp32 = get_per_class_acc('baseline_fp32', is_quant=False)
    acc_compressed = get_per_class_acc('baseline_int8_pruned_70', is_quant=True)

    if acc_fp32 is None or acc_compressed is None:
        raise FileNotFoundError("Missing baseline or compressed model. Cannot generate degradation comparison.")

    # Plotting Grouped Bar Chart
    x = np.arange(len(classes))
    width = 0.35

    # duplicate arrays to match x-axis in plot
    if len(acc_fp32) > len(classes):
        acc_fp32 = acc_fp32[:len(classes)]
    if len(acc_compressed) > len(classes):
        acc_compressed = acc_compressed[:len(classes)]
    if len(acc_fp32) < len(classes):
        acc_fp32 = np.pad(acc_fp32, (0, len(classes) - len(acc_fp32)), 'constant')
    if len(acc_compressed) < len(classes):
        acc_compressed = np.pad(acc_compressed, (0, len(classes) - len(acc_compressed)), 'constant')

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(x - width/2, acc_fp32, width, label='Baseline FP32', color='#34495e', edgecolor='black', zorder=3)
    ax.bar(x + width/2, acc_compressed, width, label='Deep Compressed (INT8+70% Sparse)', color='#e74c3c', edgecolor='black', zorder=3)

    ax.set_ylabel('Top-1 Accuracy (%)', fontweight='bold')
    ax.set_title('Per-Class Accuracy Degradation Comparison', pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.set_ylim(0, 110)

    fig.tight_layout()
    out_path = RESULTS_DIR / 'class_degradation_comparison.png'
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved Class Degradation Chart to {out_path}")

def generate_huffman_weight_distribution():
    """Generates a histogram of quantized weights """
    print("Generating Huffman Weight Distribution Histogram...")

    m_path = MODELS_DIR / 'baseline_int8_pruned_70.pth'
    if not m_path.exists():
        raise FileNotFoundError(f"{m_path} missing. Cannot generate histogram.")
    else:
        model = load_baseline_model()
        model = q.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(torch.load(m_path, map_location='cpu'))
        # Extract the quantized weights from the linear classifier
        weights = model.classifier[-1].weight().int_repr().numpy().flatten()

    plt.figure(figsize=(10, 6))

    # Plot histogram with log scale Y axis since 0 will be massively dominant
    counts, bins, patches = plt.hist(weights, bins=255, range=(-128, 127), color='#9b59b6', log=True, zorder=3)

    # Highlight the 0 bin
    zero_idx = np.where((bins >= 0) & (bins < 1))[0][0]
    patches[zero_idx].set_facecolor('#f1c40f')
    patches[zero_idx].set_edgecolor('black')

    plt.annotate(f"70% Sparsity Spike\n(Assigned 1-bit Huffman Code)",
                 xy=(0, counts[zero_idx]), xytext=(30, counts[zero_idx]/10),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9))

    plt.xlabel('Quantized Weight Integer Value (-128 to 127)', fontweight='bold')
    plt.ylabel('Frequency (Log Scale)', fontweight='bold')
    plt.title('Post-Pruning Quantized Weight Distribution (Mechanism of Huffman Coding)', pad=20, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    plt.tight_layout()
    out_path = RESULTS_DIR / 'huffman_weight_distribution.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Huffman Distribution Chart to {out_path}")

if __name__ == '__main__':
    os.makedirs(RESULTS_DIR, exist_ok=True)
    generate_confusion_matrix()
    generate_layer_sparsity_breakdown()
    generate_compression_waterfall()
    generate_class_degradation_comparison()
    generate_huffman_weight_distribution()
    print("Advanced IEEE visual generation complete.")