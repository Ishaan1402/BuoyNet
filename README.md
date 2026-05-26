# BuoyNet

Classifying marine microplastic debris using **MobileNetV3-Small**, with Deep Compression (quantization / pruning / weight coding), profiling (accuracy, latency/size/energy proxies), and **synthetic domain shift** tests in order to mimic turbid water, biofouling, and poor lighting.

**Paper:** [BuoyNet (PDF)](reference/BuoyNet.pdf)

---

## SparkNotes

- **Problem:** Classify drifting plastic fragments from buoy-mounted imagery where models must stay small and fast enough for edge hardware.
- **Approach:** MobileNetV3‑Small, trained on ImageNet, fine-tuned on a multi-class debris [dataset](https://figshare.com/articles/dataset/DeepParticle_dataset_MICRO_MESO_MACRO_2022_/26511253); compressed with quantization-aware training (INT8), L1 unstructured pruning on conv/linear weights, plus Huffman weight encoding ([compression](scripts/compression_pipeline.py)).
- **Method:** Train → QAT → compress → evaluate; scripts export efficiency and quality metrics to be used for plotting([eval](scripts/ieee_master_eval.py), [Pareto / figures](scripts/analyze_pareto.py)).
- **Robustness check:** Held-out accuracy on clean and domain-shift augmented “messy” splits ([shift generator](scripts/augment_domain_shift.py), [shift eval](scripts/evaluate_domain_shift.py)).

Stack: Python, PyTorch, torchvision, OpenCV, scikit-learn, pandas/matplotlib.

---

## Table of Contents


| For                                                                                 | Go to                                                                                                              |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Academic paper (methods, results, evaluation)                                       | [reference/BuoyNet.pdf](reference/BuoyNet.pdf)                                                                     |
| End-to-end pipeline (preprocessing → train → QAT → compress → plot)                 | [notebooks/BuoyNet_Master_Pipeline.ipynb](notebooks/BuoyNet_Master_Pipeline.ipynb)                                 |
| QAT-focused pipeline                                                                | [notebooks/BuoyNet_QAT_Pipeline.ipynb](notebooks/BuoyNet_QAT_Pipeline.ipynb)                                       |
| Evaluation + plotting                                                               | [notebooks/BuoyNet_Eval_Pipeline.ipynb](notebooks/BuoyNet_Eval_Pipeline.ipynb)                                     |
| Flatten raw dataset +`ImageFolder` splits                                           | [scripts/prepare_dataset.py](scripts/prepare_dataset.py)                                                           |
| Baseline training                                                                   | [scripts/train_baseline.py](scripts/train_baseline.py)                                                             |
| PTQ, pruning checkpoints, Huffman stats (Used in earlier stages of experimentation) | [scripts/compression_pipeline.py](scripts/compression_pipeline.py)                                                 |
| Full metric table (acc/F1/precision/recall + size/latency/energy)                   | [scripts/ieee_master_eval.py](scripts/ieee_master_eval.py)                                                         |
| Synthetic underwater degradations                                                   | [scripts/augment_domain_shift.py](scripts/augment_domain_shift.py)                                                 |
| Accuracy on clean vs. shifted test folders                                          | [scripts/evaluate_domain_shift.py](scripts/evaluate_domain_shift.py)                                               |
| Aggregate latency/size for Pareto-style charts                                      | [scripts/simulate_latency.py](scripts/simulate_latency.py), [scripts/analyze_pareto.py](scripts/analyze_pareto.py) |
| Plotting figures                                                                    | [scripts/advanced_ieee_plots.py](scripts/advanced_ieee_plots.py)                                                   |


---

## Notes

1. **No field deployment.** BuoyNet was never mounted on a buoy or run in open water. All results come from offline training, compression, and scripted evaluation. The goal is a design study at the intersection of IoT hardware and software. Given a pretrained backbone and dataset, which compressed variants can stay accurate enough, and which seem viable on paper for edge hardware?
2. **Domain shift is a proxy for real-world underwater conditions.** [augment_domain_shift.py](scripts/augment_domain_shift.py) applies blur, color cast, and lighting tweaks to approximate turbidity, biofouling, and poor illumination. Those transforms stress the model in a controlled way and expose relative drops between FP32 and compressed checkpoints but they do **not** fully validate performance on real underwater footage. The shifted splits are a  **check on the robustness of the model**, not evidence that BuoyNet can seamlessly generalize in the field.
3. **PTQ deprecated for QAT.** Early experiments used post-training quantization in [compression_pipeline.py](scripts/compression_pipeline.py) (`quantize_dynamic` → `baseline_int8_ptq.pth`) as a quick baseline. Accuracy loss was too high for the paper’s targets, so the main pipeline moved to **quantization-aware training** ([BuoyNet_QAT_Pipeline.ipynb](notebooks/BuoyNet_QAT_Pipeline.ipynb) → `qat_int8_baseline.pth`). The PTQ script stays in the repo as an ablation path; the paper's numbers and Pareto labels are derived from the QAT variants.
4. **Reproducing.** Notebooks were run on Colab with manual steps. Re-running end-to-end requires the DeepParticle dataset and GPUs for QAT.

---

## Quick run

Requirements: `torch`, `torchvision`, `opencv-python`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `Pillow`.

1. Put your raw imagery (varying sizes) under `data/raw/` following the hierarchy expected by [prepare_dataset.py](scripts/prepare_dataset.py).
2. Run prep, then baseline training, then compression/eval scripts in notebook order, or open a master notebook and execute top to bottom.
3. Inspect `figures/` and `logs/` for exported CSVs and plots.

CUDA optional; quantized eval paths expect CPU in parts of PyTorch’s dynamic quantization path.