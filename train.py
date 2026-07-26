import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModel, CLIPVisionModel,
    TrainingArguments, Trainer, TrainerCallback, set_seed
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def seed_everything(seed=0):
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
    "w_t": 1.0,
    "w_i": 1.0,
    "w_c": 1.0,
    "weight_cl": 1.0,
    "weight_avu": 1.0,
    "dpcr_lambda": 0.1,
}

seed_everything(CONFIG["seed"])

dpcr_lambda = CONFIG["dpcr_lambda"]
DPCR_WARMUP_START_EPOCH = 1.0
DPCR_WARMUP_END_EPOCH = 3.0


def get_dpcr_lambda_eff(current_epoch):
    if current_epoch is None:
        current_epoch = 0.0
    current_epoch = float(current_epoch)
    if current_epoch <= DPCR_WARMUP_START_EPOCH:
        warmup_factor = 0.0
    elif current_epoch >= DPCR_WARMUP_END_EPOCH:
        warmup_factor = 1.0
    else:
        warmup_factor = (
                (current_epoch - DPCR_WARMUP_START_EPOCH)
                / (DPCR_WARMUP_END_EPOCH - DPCR_WARMUP_START_EPOCH + 1e-8)
        )
    return dpcr_lambda * warmup_factor


OUTPUT_DIR = "./output"
TEXT_MODEL_NAME = "./models/DeBERTa-v2-710M-Chinese"
IMAGE_MODEL_NAME = "./models/chinese-clip-vit-large-patch14"
TRAIN_CSV = "./data/train.csv"
TRAIN_PKL = "./data/train_clip_loader.pkl"
VAL_CSV = "./data/val.csv"
VAL_PKL = "./data/val_clip_loader.pkl"

MAX_LENGTH = 512
BATCH_SIZE = 8
GRADIENT_ACC_STEPS = 4
NUM_LABELS = 2
NUM_EPOCHS = 8
LR_BACKBONE = 1e-5
LR_HEAD = 2e-4


def get_composite_uncertainty(evidence, num_classes=2):
    S = torch.sum(evidence + 1, dim=1, keepdim=True)
    u_vac = num_classes / S
    p = (evidence + 1) / S
    H = -torch.sum(p * torch.log(p + 1e-8), dim=1, keepdim=True)
    u_ent = H / np.log(num_classes)
    return 0.5 * u_vac + 0.5 * u_ent


def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step):
    evidence = F.softplus(output)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    y_oh = target
    loss_ace = torch.sum(y_oh * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    coef = torch.min(
        torch.tensor(1.0, device=output.device),
        torch.tensor(epoch_num / annealing_step, device=output.device)
    )
    gamma = torch.lgamma
    ones = torch.ones([1, num_classes], device=output.device)
    kl = (
                 gamma(torch.sum(alpha, dim=1, keepdim=True))
                 - gamma(alpha).sum(dim=1, keepdim=True)
                 + gamma(ones).sum(dim=1, keepdim=True)
                 - gamma(ones.sum(dim=1, keepdim=True))
         ) + (
             (alpha - ones)
             .mul(
                 torch.digamma(alpha)
                 - torch.digamma(torch.sum(alpha, dim=1, keepdim=True))
             )
             .sum(dim=1, keepdim=True)
         )
    return torch.mean(loss_ace + coef * kl)


def avu_loss(logits, labels):
    evidence = F.softplus(logits)
    S = torch.sum(evidence + 1, dim=1, keepdim=True)
    u = get_composite_uncertainty(evidence, 2)
    preds = torch.argmax((evidence + 1) / S, dim=1)
    reward = (1 - u[preds == labels]).sum() + u[preds != labels].sum()
    return -torch.log((reward + 1e-10) / logits.size(0))


def three_branch_anchor_fusion(logits_t, logits_i, logits_c, num_classes=2, current_epoch=0.0):
    e_t = F.softplus(logits_t)
    e_i = F.softplus(logits_i)
    e_c = F.softplus(logits_c)

    S_t, S_i, S_c = e_t + 1, e_i + 1, e_c + 1
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

    dpcr_lambda_eff = get_dpcr_lambda_eff(current_epoch)

    b_final_0 = b_base_0 + C_total * r_base_0
    b_final_1 = b_base_1 + C_total * r_base_1

    delta_0 = (r_cross_0 - r_base_0).detach()
    delta_1 = (r_cross_1 - r_base_1).detach()

    b_final_0 = b_final_0 + dpcr_lambda_eff * C_micro_eff * delta_0
    b_final_1 = b_final_1 + dpcr_lambda_eff * C_micro_eff * delta_1

    u_final = w_t_base * u_t + w_i_base * u_i + w_c_base * u_c
    total_mass = b_final_0 + b_final_1 + u_final

    probs = torch.cat([b_final_0, b_final_1], dim=1) / (total_mass + 1e-8)

    return probs, C_total


class MultimodalV6_ThreeBranch(nn.Module):
    def __init__(self, text_model_name, img_model_name):
        super().__init__()
        self.current_epoch = 0.0

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

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=4,
            batch_first=True,
            dropout=0.1
        )

        self.cross_proj = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.LayerNorm(text_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(text_dim // 2, NUM_LABELS)
        )

        self._freeze_backbones()

        self.text_encoder.gradient_checkpointing_enable()
        self.text_encoder.enable_input_require_grads()

    def _freeze_backbones(self):
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for l in self.text_encoder.encoder.layer[-6:]:
            for p in l.parameters():
                p.requires_grad = True
        for p in self.img_encoder.parameters():
            p.requires_grad = False
        for l in self.img_encoder.vision_model.encoder.layers[-6:]:
            for p in l.parameters():
                p.requires_grad = True

    def forward(self, input_ids, attention_mask, pixel_values, labels=None, **kwargs):
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

        probs, total_conflict = three_branch_anchor_fusion(
            l_text,
            img_fusion_input,
            l_cross,
            NUM_LABELS,
            self.current_epoch
        )

        return SequenceClassifierOutput(
            logits=probs,
            hidden_states=(total_conflict, l_text, l_img, l_cross, t_norm, i_norm)
        )


class PixelTensorDataset(Dataset):
    def __init__(self, csv_path, pkl_path, tokenizer, max_len=512):
        self.df = pd.read_csv(csv_path)
        with open(pkl_path, "rb") as f:
            self.img_tensor = pickle.load(f)
        self.length = min(len(self.df), len(self.img_tensor))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row.get("content") or row.get("text") or "")
        label = int(row.get("label", 0))
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "pixel_values": self.img_tensor[idx].clone(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class FinalEvalCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        control.should_evaluate = True
        return control


class ThreeBranchTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        target_model = model.module if hasattr(model, "module") else model
        target_model.current_epoch = float(self.state.epoch or 0.0)

        outputs = model(**inputs)

        probs = outputs.logits
        l_text = outputs.hidden_states[1]
        l_img = outputs.hidden_states[2]
        l_cross = outputs.hidden_states[3]
        t_norm = outputs.hidden_states[4]
        i_norm = outputs.hidden_states[5]

        y_oh = F.one_hot(labels, 2).float()
        epoch = self.state.epoch or 0

        w_text = CONFIG["w_t"] * (1.0 - get_composite_uncertainty(F.softplus(l_text.detach()), 2)).mean()
        w_img = CONFIG["w_i"] * (1.0 - get_composite_uncertainty(F.softplus(l_img.detach()), 2)).mean()
        w_cross = CONFIG["w_c"] * (1.0 - get_composite_uncertainty(F.softplus(l_cross.detach()), 2)).mean()

        loss_main = F.nll_loss(torch.log(probs + 1e-8), labels)

        loss_text = w_text * edl_digamma_loss(l_text, y_oh, epoch, 2, NUM_EPOCHS)
        loss_img = w_img * edl_digamma_loss(l_img, y_oh, epoch, 2, NUM_EPOCHS)
        loss_cross = w_cross * edl_digamma_loss(l_cross, y_oh, epoch, 2, NUM_EPOCHS)

        l_avu = CONFIG["weight_avu"] * avu_loss(l_text + l_img + l_cross, labels)

        curr_batch_size = t_norm.size(0)
        logit_scale = 1.0 / 0.07
        logits_per_text = logit_scale * t_norm @ i_norm.t()
        logits_per_image = logit_scale * i_norm @ t_norm.t()
        ce_labels = torch.arange(curr_batch_size, device=t_norm.device)
        base_cl_loss = (
                               F.cross_entropy(logits_per_text, ce_labels, reduction='none')
                               + F.cross_entropy(logits_per_image, ce_labels, reduction='none')
                       ) / 2.0
        cl_weights = torch.where(labels == 0, 1.0, 0.1)
        loss_cl = CONFIG["weight_cl"] * (base_cl_loss * cl_weights).mean()

        total_loss = loss_main + loss_text + loss_img + loss_cross + l_avu + loss_cl

        return (total_loss, outputs) if return_outputs else total_loss


def compute_metrics(p):
    preds = np.argmax(p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(p.label_ids, preds), "f1": f1}


if __name__ == "__main__":
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    train_ds = PixelTensorDataset(TRAIN_CSV, TRAIN_PKL, tokenizer, MAX_LENGTH)
    val_ds = PixelTensorDataset(VAL_CSV, VAL_PKL, tokenizer, MAX_LENGTH)
    model = MultimodalV6_ThreeBranch(TEXT_MODEL_NAME, IMAGE_MODEL_NAME)

    head_params = (
            list(map(id, model.text_proj.parameters()))
            + list(map(id, model.img_proj.parameters()))
            + list(map(id, model.cross_proj.parameters()))
            + list(map(id, model.sim_proj.parameters()))
            + list(map(id, model.img_seq_proj.parameters()))
            + list(map(id, model.cross_attn.parameters()))
    )

    base_params = filter(lambda p: id(p) not in head_params and p.requires_grad, model.parameters())

    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": LR_BACKBONE},
        {"params": model.text_proj.parameters(), "lr": LR_HEAD},
        {"params": model.img_proj.parameters(), "lr": LR_HEAD},
        {"params": model.cross_proj.parameters(), "lr": LR_HEAD},
        {"params": model.sim_proj.parameters(), "lr": LR_HEAD},
        {"params": model.img_seq_proj.parameters(), "lr": LR_HEAD},
        {"params": model.cross_attn.parameters(), "lr": LR_HEAD},
    ])

    training_args_dict = {
        "output_dir": OUTPUT_DIR,
        "seed": CONFIG["seed"],
        "data_seed": CONFIG["seed"],
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACC_STEPS,
        "bf16": True,
        "warmup_ratio": 0.1,
        "eval_strategy": "steps",
        "eval_steps": 50,
        "save_strategy": "steps",
        "save_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 8,
        "report_to": "none",
        "logging_steps": 10,
        "max_grad_norm": 1.0,
        "remove_unused_columns": False,
    }

    train_args = TrainingArguments(**training_args_dict)

    trainer = ThreeBranchTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=[FinalEvalCallback()]
    )

    trainer.train()