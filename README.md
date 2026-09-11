# ECAM: Multimodal Fake News Detection via Evidence Conflict-Aware Management

This repository contains the official PyTorch implementation of **ECAM: Multimodal Fake News Detection via Evidence Conflict-Aware Management**.

## 📖 Introduction

Text and images in a news post may provide different judgments about its authenticity. Such disagreement can contain useful discriminative information, motivating explicit modeling of how evidence from different sources supports competing classes.

ECAM models this disagreement in the class belief space. It extracts evidence from textual, visual, and cross-modal interaction branches, quantifies their conflicts using textual belief as an anchor, and incorporates conflict magnitude and class direction into the final decision through proportional reallocation and residual directional correction.

The framework is evaluated on **Weibo-16**, **Weibo-21**, and **Twitter**.

## ✨ Key Features

- **Evidential representation and composite uncertainty.** The three branches produce class evidence and belief masses. Composite uncertainty combines evidential vacuity with normalized predictive entropy to guide adaptive fusion.
- **Text-anchored conflict quantification.** Macroscopic conflict measures disagreement between textual and visual beliefs, while microscopic conflict measures disagreement between textual and cross-modal interaction beliefs.
- **Conflict-informed class support updating.** Main-path proportional reallocation uses conflict magnitude to adjust class support, while residual directional correction incorporates class direction from the interaction branch.
- **End-to-end evidence learning.** The final class support scores provide the main classification supervision, allowing conflict management to influence upstream evidence learning during training. Branch beliefs, uncertainty estimates, and conflict measures also support analysis of model behavior.

## 📂 Project Structure

| Path | Description |
| --- | --- |
| `data/train.csv` | Training texts and labels |
| `data/train_clip_loader.pkl` | Preprocessed image tensors aligned with the training CSV |
| `data/val.csv` | Validation texts and labels |
| `data/val_clip_loader.pkl` | Preprocessed image tensors aligned with the validation CSV |
| `data/test.csv` | Test texts and labels |
| `data/test_clip_loader.pkl` | Preprocessed image tensors aligned with the test CSV |
| `models/` | Local pre-trained encoder, tokenizer, and processor files |
| `output/` | Suggested directory for training outputs and checkpoints |
| `train.py` | Training script |
| `test.py` | Evaluation script |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git ignore rules |
| `README.md` | Project documentation |

Prepare the CSV and PKL files for the selected dataset and set their paths in the scripts. Separate data and output directories can be used for different datasets and experiment runs.

## 📥 Datasets and Data Preparation

### Dataset Sources

| Dataset | Language | Access |
| --- | --- | --- |
| Weibo-16 | Chinese | [Processed data on Baidu Netdisk](https://pan.baidu.com/s/1IkrSGc8IW9TQDmkMD6-LTw?pwd=b6f4), access code: `b6f4` |
| Weibo-21 | Chinese | [MDFEND-Weibo21](https://github.com/kennqiang/MDFEND-Weibo21); follow the repository's instructions for access to the original dataset |
| Twitter | English | [Image Verification Corpus](https://github.com/MKLab-ITI/image-verification-corpus), including the MediaEval 2015 data |

The source repositories provide their own data formats. Prepare the CSV/PKL pairs described below for the ECAM data loader, using the corresponding experimental split.

### Input Format

Each split consists of a CSV file and a matching PKL file.

**CSV fields:**

| Field | Description |
| --- | --- |
| `content` | News text; `text` is also supported as an alternative column name |
| `label` | Integer class label: `0` for real news and `1` for fake news |

**PKL contents:**

Each `*_clip_loader.pkl` file stores preprocessed image tensors. The loader retrieves the tensor at the same row index as the text and passes it to the image encoder as `pixel_values`. Each sample should provide an image tensor of shape `(3, 224, 224)`, prepared using the processor associated with the selected image encoder.

**The PKL files contain image inputs, not pre-extracted CLIP feature embeddings.** The CSV and PKL files must have the same number of samples and identical sample ordering. Keep their ordering synchronized when filtering or splitting the data.

## 🧠 Pre-trained Encoders

ECAM uses language-specific **DeBERTa-family text encoders** and **CLIP-family image encoders**.

| Datasets | Component | Pre-trained Model |
| --- | --- | --- |
| Weibo-16 / Weibo-21 | Text | [IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese) |
| Weibo-16 / Weibo-21 | Image | [OFA-Sys/chinese-clip-vit-large-patch14](https://huggingface.co/OFA-Sys/chinese-clip-vit-large-patch14) |
| Twitter | Text | [microsoft/deberta-v2-xlarge](https://huggingface.co/microsoft/deberta-v2-xlarge) |
| Twitter | Image | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) |

Download the models needed for the selected dataset, including their configuration and tokenizer or processor files. The corresponding local directory names are:

- `models/DeBERTa-v2-710M-Chinese/`
- `models/chinese-clip-vit-large-patch14/`
- `models/deberta-v2-xlarge/`
- `models/clip-vit-large-patch14/`

Other directory names can be used by updating the model paths in the scripts.

| Setting | Weibo-16 / Weibo-21 | Twitter |
| --- | --- | --- |
| Maximum text length | 512 tokens | 128 tokens |
| Image input size | 224 × 224 | 224 × 224 |

## 🛠️ Environment Setup

Use a Python environment compatible with the dependencies in `requirements.txt`. The provided training and evaluation setup uses CUDA; configure PyTorch for your GPU environment.

### 1. Install PyTorch

Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) to choose the installation command for your operating system and CUDA environment.

### 2. Install the Remaining Dependencies

Run the following command from the project root:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Configure an Experiment

Update the configuration section of `train.py` and `test.py` for the selected dataset. The main settings include:

- CSV and PKL paths for each split.
- Text and image encoder paths.
- Maximum text length, batch size, and output directory.
- Random seed, learning rates, loss weights, and `dpcr_lambda`.
- Checkpoint path and model variant for evaluation.

Keep the preprocessing, encoder choices, and model variant consistent between training and evaluation. If residual warm-up is used, evaluation should use the effective residual coefficient associated with the selected checkpoint.

### 2. Train

Run the following command after configuring the training script:

```bash
python train.py
```

Training starts from the downloaded pre-trained encoders. Checkpoints are saved to the configured output directory, and the best checkpoint is selected by **validation F1 for the fake-news class** (`label = 1`).

### 3. Evaluate

Set the checkpoint and test-data paths in `test.py`, then run:

```bash
python test.py
```

The evaluation script computes classification metrics, including accuracy, macro-F1, and class-wise precision, recall, and F1.

The model outputs **class support scores** for classification. These scores are used to compare classes and are not interpreted as normalized class probabilities. Predictions are obtained by selecting the class with the highest support score.

## 📊 Results Reported in the Manuscript

| Dataset | Accuracy | Macro-F1 | Fake News F1 | Real News F1 |
| --- | ---: | ---: | ---: | ---: |
| Weibo-16 | 0.9604 | 0.9604 | 0.9617 | 0.9590 |
| Weibo-21 | 0.9532 | 0.9532 | 0.9532 | 0.9531 |
| Twitter | 0.9576 | 0.9572 | 0.9613 | 0.9530 |

These values are the test results reported in the manuscript under the corresponding experimental configurations.

The manuscript also includes ablation studies, uncertainty and prediction-risk analysis, and conflict-stratified comparisons. The latter compare ECAM with the separately trained variant without conflict management on shared test groups defined by ECAM's conflict-score tertiles. The largest accuracy gains occur in the high-conflict groups across all three datasets.

## 🙏 Acknowledgments

We thank the developers of DeBERTa, CLIP, Chinese-CLIP, and the dataset providers for making their research resources available.
