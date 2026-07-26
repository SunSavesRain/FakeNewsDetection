# ECAM: Multimodal Fake News Detection via Evidence Conflict-Aware Management

This repository contains the official PyTorch implementation of the paper **"ECAM: Multimodal Fake News Detection via Evidence Conflict-Aware Management"**. 

## 📖 Introduction

As social media becomes the mainstream way of information dissemination, the derived methods of news fabrication have further exacerbated the harm to network and public security. Although existing automatic multimodal fake news detection methods have achieved initial success, they still face several limitations: 
1. It is difficult to directly quantify the conflict between modalities.
2. The conflict between modalities cannot be utilized reasonably.
3. The models are too black-box, resulting in poor interpretability.

To address these issues, we propose **ECAM**, a multimodal fake news detection model based on evidence conflict-aware management. 

## ✨ Key Features

* **Subjective Logic & Evidence Theory:** ECAM maps text, visual, and cross-modal interaction features into category belief masses and cognitive uncertainty. This enables the model to characterize the support degree of different information sources for news authenticity in the evidence space.
* **Evidence Conflict Quantification:** The model establishes the text modality as the core belief anchor. It measures the evidence divergence of different modalities in the category belief space from two dimensions: single-modal independent discriminative evidence and cross-modal fine-grained interaction evidence.
* **Conflict Management Mechanism:** The quantified cross-modal evidence conflict is used as a discriminative belief increment for category conditions. It is injected into the final decision process through main-path proportional redistribution and micro-conflict residual directional correction, allowing the model to explicitly utilize the evidence divergence between different modalities.

## 📂 Project Structure

```text
FakeNewsDetection/
├── data/
│   ├── train.csv                 # Training text and labels
│   ├── train_clip_loader.pkl     # Pre-extracted CLIP visual features for training
│   ├── val.csv                   # Validation text and labels
│   ├── val_clip_loader.pkl       # Pre-extracted CLIP visual features for validation
│   ├── test.csv                  # Test text and labels
│   └── test_clip_loader.pkl      # Pre-extracted CLIP visual features for testing
├── models/
│   ├── DeBERTa-v2-710M-Chinese   # Local weights for the text encoder
│   └── chinese-clip-vit-large-patch14 # Local weights for the image encoder
├── output/                       # Output directory for saved checkpoints
├── train.py                      # Main training script
├── test.py                       # Evaluation script
├── requirements.txt              # Environment dependencies
├── .gitignore                    # Git ignore configuration
└── README.md                     # This document
```

## 📥 Datasets & Pre-trained Models

### Datasets
* **Weibo-16:** We provide the processed Weibo-16 dataset via Baidu Netdisk. You can download it here: https://pan.baidu.com/s/1IkrSGc8IW9TQDmkMD6-LTw?pwd=b6f4. 
* **Weibo-21:** The Weibo-21 dataset can be found in its official repository: [MDFEND-Weibo21 on GitHub](https://github.com/kennqiang/MDFEND-Weibo21).

### Pre-trained Models
Please download the following pre-trained weights from Hugging Face and place them in the `./models/` directory (or modify the paths in the `CONFIG` of our scripts):
* **Text Encoder:** [IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese)
* **Image Encoder:** [OFA-Sys/chinese-clip-vit-large-patch14](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14)

## 🛠️ Environment Setup

Please ensure you have Python 3.8+ installed.

**1. Install PyTorch**  
Since CUDA support varies across different hardware configurations, please visit the [official PyTorch website](https://pytorch.org/get-started/locally/) to get and run the appropriate installation command for your specific GPU environment.

**2. Install Other Dependencies**  
After configuring PyTorch, please run the following command in the root directory of the project to install all remaining dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

Hyperparameters (such as loss weights, `seed`, and `dpcr_lambda`) are centrally managed in the `CONFIG` dictionary at the top of the `train.py` and `test.py` scripts. You can easily tune them for your specific environment.

### 1. Training
To train the ECAM model from scratch, simply run:
```bash
python train.py
```
The best checkpoint will be automatically saved in the `./output/` directory by default based on the validation F1-score.

### 2. Evaluation
To evaluate the trained model on the test set, specify your checkpoint path in the `CONFIG` of `test.py` and run:
```bash
python test.py
```
This will output a detailed evaluation report including Overall Metrics (Accuracy, Macro-F1) and Fake News Metrics.
