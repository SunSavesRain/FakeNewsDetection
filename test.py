import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pickle
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, CLIPVisionModel, set_seed
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    set_seed(seed)

CONFIG = {
    "seed": 0,
    "dpcr_lambda": 0.1,
    "checkpoint_path": "./output/checkpoint-best",
}

seed_everything(CONFIG["seed"])

dpcr_lambda = CONFIG["dpcr_lambda"]

TEXT_MODEL_NAME = "./models/DeBERTa-v2-710M-Chinese"
IMAGE_MODEL_NAME = "./models/chinese-clip-vit-large-patch14"
TEST_CSV = "./data/test.csv"
TEST_PKL = "./data/test_clip_loader.pkl"
NUM_LABELS = 2
MAX_LENGTH = 512
BATCH_SIZE = 16

def get_composite_uncertainty(evidence, num_classes=2):
    S = torch.sum(evidence + 1, dim=1, keepdim=True)
    u_vac = num_classes / S
    p = (evidence + 1) / S
    H = -torch.sum(p * torch.log(p + 1e-8), dim=1, keepdim=True)
    u_ent = H / np.log(num_classes)
    return 0.5 * u_vac + 0.5 * u_ent

def three_branch_anchor_fusion(logits_t, logits_i, logits_c, num_classes=2):
    e_t = F.softplus(logits_t)
    e_i = F.softplus(logits_i)
    e_c = F.softplus(logits_c)

    S_t = e_t + 1
    S_i = e_i + 1
    S_c = e_c + 1

    S_t_sum = S_t.sum(1, keepdim=True)
    S_i_sum = S_i.sum(1, keepdim=True)
    S_c_sum = S_c.sum(1, keepdim=True)

    b_t = e_t / S_t_sum
    b_i = e_i / S_i_sum
    b_c = e_c / S_c_sum

    u_t = get_composite_uncertainty(e_t, num_classes)
    u_i = get_composite_uncertainty(e_i, num_classes)
    u_c = get_composite_uncertainty(e_c, num_classes)

    C_macro = (b_t[:, 0:1] * b_i[:, 1:2]) + (b_t[:, 1:2] * b_i[:, 0:1])
    C_micro = (b_t[:, 0:1] * b_c[:, 1:2]) + (b_t[:, 1:2] * b_c[:, 0:1])

    w_macro = (1.0 - u_i) / (2.0 - u_i - u_c + 1e-8)
    w_micro = (1.0 - u_c) / (2.0 - u_i - u_c + 1e-8)

    C_macro_eff = w_macro * C_macro
    C_micro_eff = w_micro * C_micro
    C_total = C_macro_eff + C_micro_eff

    denom = u_t * u_i + u_i * u_c + u_c * u_t + 1e-8
    w_t_base = (u_i * u_c) / denom
    w_i_base = (u_t * u_c) / denom
    w_c_base = (u_t * u_i) / denom

    b_base_0 = (w_t_base * b_t[:, 0:1] + w_i_base * b_i[:, 0:1] + w_c_base * b_c[:, 0:1])
    b_base_1 = (w_t_base * b_t[:, 1:2] + w_i_base * b_i[:, 1:2] + w_c_base * b_c[:, 1:2])

    norm_s = b_base_0 + b_base_1 + 1e-8

    r_base_0 = b_base_0 / norm_s
    r_base_1 = b_base_1 / norm_s

    cross_s = b_c[:, 0:1] + b_c[:, 1:2] + 1e-8
    r_cross_0 = b_c[:, 0:1] / cross_s
    r_cross_1 = b_c[:, 1:2] / cross_s

    b_final_0 = b_base_0 + C_total * r_base_0
    b_final_1 = b_base_1 + C_total * r_base_1

    delta_0 = (r_cross_0 - r_base_0).detach()
    delta_1 = (r_cross_1 - r_base_1).detach()

    b_final_0 = b_final_0 + dpcr_lambda * C_micro_eff * delta_0
    b_final_1 = b_final_1 + dpcr_lambda * C_micro_eff * delta_1

    u_final = w_t_base * u_t + w_i_base * u_i + w_c_base * u_c
    total_mass = b_final_0 + b_final_1 + u_final

    probs = torch.cat([b_final_0, b_final_1], dim=1) / (total_mass + 1e-8)

    return probs

class MultimodalV6_ThreeBranch(nn.Module):
    def __init__(self, text_model_name, img_model_name):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.img_encoder = CLIPVisionModel.from_pretrained(img_model_name)
        text_dim = self.text_encoder.config.hidden_size
        img_dim = self.img_encoder.config.hidden_size

        self.sim_proj = nn.Linear(text_dim, img_dim) if text_dim != img_dim else nn.Identity()

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.LayerNorm(text_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(text_dim // 2, NUM_LABELS)
        )

        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, img_dim // 2),
            nn.LayerNorm(img_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(img_dim // 2, NUM_LABELS)
        )

        self.img_seq_proj = nn.Linear(img_dim, text_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=text_dim, num_heads=4, batch_first=True, dropout=0.1)

        self.cross_proj = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.LayerNorm(text_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(text_dim // 2, NUM_LABELS)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        text_seq = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_cls = text_seq[:, 0, :]

        img_out = self.img_encoder(pixel_values=pixel_values)
        img_seq, img_pool = img_out.last_hidden_state, img_out.pooler_output

        img_seq_aligned = self.img_seq_proj(img_seq)
        cross_seq, _ = self.cross_attn(query=text_seq, key=img_seq_aligned, value=img_seq_aligned)
        cross_cls = cross_seq[:, 0, :]

        l_text = self.text_proj(text_cls)
        l_img = self.img_proj(img_pool)
        l_cross = self.cross_proj(cross_cls)

        text_emb_aligned = self.sim_proj(text_cls)
        t_norm = F.normalize(text_emb_aligned, p=2, dim=1)
        i_norm = F.normalize(img_pool, p=2, dim=1)
        gate = torch.sigmoid((torch.sum(t_norm * i_norm, dim=1, keepdim=True) + 0.5) * 5)

        img_fusion_input = F.softplus(l_img) * gate

        probs = three_branch_anchor_fusion(l_text, img_fusion_input, l_cross, NUM_LABELS)

        return probs

class PixelTensorDataset(Dataset):
    def __init__(self, csv_path, pkl_path, tokenizer, max_len=512):
        self.df = pd.read_csv(csv_path)
        with open(pkl_path, 'rb') as f:
            self.img_tensor = pickle.load(f)
        self.length = min(len(self.df), len(self.img_tensor))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row.get('content') or row.get('text') or "")
        label = int(row.get('label', 0))
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'pixel_values': self.img_tensor[idx].clone(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = MultimodalV6_ThreeBranch(TEXT_MODEL_NAME, IMAGE_MODEL_NAME)

    safe_path = os.path.join(CONFIG["checkpoint_path"], "model.safetensors")
    bin_path = os.path.join(CONFIG["checkpoint_path"], "pytorch_model.bin")

    if os.path.exists(safe_path):
        from safetensors.torch import load_file
        state_dict = load_file(safe_path, device="cpu")
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Missing weights at {CONFIG['checkpoint_path']}")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.cuda()

    test_ds = PixelTensorDataset(TEST_CSV, TEST_PKL, tokenizer, max_len=MAX_LENGTH)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            pixel_values = batch['pixel_values'].cuda()
            labels = batch['labels']

            probs = model(input_ids, attention_mask, pixel_values)

            all_probs.append(probs.cpu())
            all_labels.append(labels)

    final_probs = torch.cat(all_probs, dim=0)
    preds = torch.argmax(final_probs, dim=1).numpy()
    labels_arr = torch.cat(all_labels, dim=0).numpy()

    acc = accuracy_score(labels_arr, preds)

    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        labels_arr, preds, average=None, labels=[0, 1], zero_division=0
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels_arr, preds, average="macro", zero_division=0
    )

    fake_p, fake_r, fake_f1, _ = precision_recall_fscore_support(
        labels_arr, preds, average="binary", pos_label=1, zero_division=0
    )

    idx_real = 0
    idx_fake = 1

    print("\n" + "=" * 65)
    print("Evaluation Results")
    print("=" * 65)
    print("Overall Metrics:")
    print(f"Accuracy      : {acc:.5f}")
    print(f"Macro-Prec    : {macro_p:.5f}")
    print(f"Macro-Recall  : {macro_r:.5f}")
    print(f"Macro-F1      : {macro_f1:.5f}")
    print("-" * 65)
    print("Fake News Metrics:")
    print(f"Fake-Prec     : {fake_p:.5f}")
    print(f"Fake-Recall   : {fake_r:.5f}")
    print(f"Fake-F1       : {fake_f1:.5f}")
    print("-" * 65)
    print("Detailed Class Metrics:")
    print(f"{'Category':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Samples'}")
    print("-" * 65)
    print(f"{'Real News (0)':<15} | {precision_arr[idx_real]:<10.4f} | {recall_arr[idx_real]:<10.4f} | {f1_arr[idx_real]:<10.4f} | {support_arr[idx_real]}")
    print(f"{'Fake News (1)':<15} | {precision_arr[idx_fake]:<10.4f} | {recall_arr[idx_fake]:<10.4f} | {f1_arr[idx_fake]:<10.4f} | {support_arr[idx_fake]}")
    print("=" * 65 + "\n")