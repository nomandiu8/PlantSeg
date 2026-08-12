---
title: PlantSeg Decision Support
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "5.31.0"
app_file: app.py
pinned: true
license: mit
---

# 🌿 PlantSeg Faithfulness-Gated Decision Support

A live deployment of the inference pipeline from the paper:

> **"Faithfulness-Gated Decision Support for In-the-Wild Plant Disease Diagnosis"**

## What it does

Upload a photo of a plant leaf or fruit, and the tool returns:

1. **Predicted disease class** — from a ConvNeXtV2-Tiny classifier
2. **Lesion severity estimate** — percentage of leaf/fruit area affected (from DeepLabV3+/EfficientNet-B3 segmenter)
3. **Confidence flag** — a ground-truth-free self-consistency check: does the classifier's own Grad-CAM attention fall inside its predicted lesion region?

## Models

| Component | Architecture | Purpose |
|---|---|---|
| Classifier | ConvNeXtV2-Tiny | Disease class prediction |
| Segmenter | DeepLabV3+ / EfficientNet-B3 | Binary lesion mask |
| Attribution | Grad-CAM | Classifier attention map |

## Important Note

This pipeline uses **only** the CNN classifier + CNN segmenter pair. The paper found that transformer-based segmenters (e.g., SegFormer-B2) do not produce faithful attention maps for this task (CAM-GT IoU 0.067 vs 0.434), so the confidence flag is validated only for this specific model combination.

## How to Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Place your model weights (`ConvNeXtV2Tiny_best.pt` and `DeepLabV3Plus_efficientnet-b3.pt`) in the same directory as `app.py`, or set the `HF_MODEL_REPO` environment variable to auto-download from Hugging Face Hub.
