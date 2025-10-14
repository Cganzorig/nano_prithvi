from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import re

import torchvision.transforms.functional as TF
import os
import rasterio
from PIL import Image

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt




@dataclass
class Config:
    image_size: int = 224
    patch_size: int = 14
    in_channels: int = 6

    n_layer: int = 32
    n_head: int = 16
    hidden_dim: int = 1280
    n_class: int = 2
    dropout: float = 0.1
    bias: bool = True

config = Config()

# ------------------ 3D Sin/Cos Positional Embedding ------------------
def get_3d_sincos_pos_embed(embed_dim, grid_size, t_size=1, cls_token=True):
    def get_1d_pos_embed(size, dim):
        pos = np.arange(size)
        omega = np.arange(dim//2) / (dim/2)
        omega = 1. / (10000**omega)
        out = np.einsum('p,d->pd', pos, omega)
        emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
        return emb

    t_emb = get_1d_pos_embed(t_size, embed_dim)
    h_emb = get_1d_pos_embed(grid_size, embed_dim)
    w_emb = get_1d_pos_embed(grid_size, embed_dim)

    pos_embed = (t_emb[:, None, None, :] +
                 h_emb[None, :, None, :] +
                 w_emb[None, None, :, :])

    pos_embed = pos_embed.reshape(-1, embed_dim)

    if cls_token:
        cls = np.zeros((1, embed_dim))
        pos_embed = np.vstack((cls, pos_embed))

    return torch.tensor(pos_embed, dtype=torch.float32).unsqueeze(0)

# ------------------ Patch Embeddings ------------------
class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Conv3d(
            in_channels=config.in_channels,
            out_channels=config.hidden_dim,
            kernel_size=(1, config.patch_size, config.patch_size),
            stride=(1, config.patch_size, config.patch_size)
        )

    def forward(self, x):
        x = self.projection(x)  # [B, hidden_dim, 1, H', W']
        x = x.squeeze(2)        # Remove time dimension: [B, hidden_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N_patches, hidden_dim]
        return x

# ------------------ Embeddings ------------------
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        grid_size = config.image_size // config.patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.patch_embed = PatchEmbeddings(config)
        self.position_embeddings = get_3d_sincos_pos_embed(
            config.hidden_dim, grid_size=grid_size, t_size=1, cls_token=True
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings.to(x.device)
        return self.dropout(x)

# --load_prithvi_encoder(model, state_dict)---------------- MLP ------------------
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)

# ------------------ Encoder Block ------------------
class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.n_head = config.n_head
        self.head_dim = config.hidden_dim // config.n_head

        self.norm1 = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.qkv = nn.Linear(config.hidden_dim, config.hidden_dim * 3, bias=True)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)

        self.norm2 = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.mlp = MLP(config)

    def forward(self, x):
        B, N, D = x.shape

        x_norm = self.norm1(x)

        qkv = self.qkv(x_norm)  # [B, N, 3*D]
        qkv = qkv.view(B, N, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is [B, n_head, N, head_dim]

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        attn = attn.transpose(1, 2).reshape(B, N, D)

        x = x + self.proj(attn)

        x_norm2 = self.norm2(x)
        x = x + self.mlp(x_norm2)
        return x


# ------------------ Vision Transformer ------------------
class VIT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = Embeddings(config)
        self.encoder = nn.Sequential(OrderedDict([
            (f"blocks_{i}", EncoderBlock(config)) for i in range(config.n_layer)
        ]))
        self.norm = nn.LayerNorm(config.hidden_dim, eps=1e-6)
        self.head = nn.Linear(config.hidden_dim, config.n_class)

    def forward(self, x, return_features=False):
        x = self.embeddings(x)  # [B, N+1, D]
        for block in self.encoder:
            x = block(x)
        x = self.norm(x)
        if return_features:
            return x
        cls_output = x[:, 0]
        out = self.head(cls_output)
        return out


def load_prithvi_encoder(custom_model, prithvi_sd):
    custom_sd = custom_model.state_dict()
    new_sd = {}

    for k in custom_sd.keys():
        new_k = k

        # Embeddings mapping (fixed naming)
        if "embeddings.cls_token" in k:
            new_k = "encoder.cls_token"
        elif "embeddings.position_embeddings" in k:
            new_k = "encoder.pos_embed"
        elif "embeddings.patch_embed.projection.weight" in k:
            new_k = "encoder.patch_embed.proj.weight"
        elif "embeddings.patch_embed.projection.bias" in k:
            new_k = "encoder.patch_embed.proj.bias"

        # Encoder blocks
        elif "encoder.blocks_" in k:
            match = re.search(r'encoder\.blocks_(\d+)\.(.*)', k)
            idx, rest = match.group(1), match.group(2)

            if "qkv.weight" in rest:
                new_k = f"encoder.blocks.{idx}.attn.qkv.weight"
            elif "qkv.bias" in rest:
                new_k = f"encoder.blocks.{idx}.attn.qkv.bias"
            elif "proj.weight" in rest:
                new_k = f"encoder.blocks.{idx}.attn.proj.weight"
            elif "proj.bias" in rest:
                new_k = f"encoder.blocks.{idx}.attn.proj.bias"
            elif "norm1.weight" in rest:
                new_k = f"encoder.blocks.{idx}.norm1.weight"
            elif "norm1.bias" in rest:
                new_k = f"encoder.blocks.{idx}.norm1.bias"
            elif "norm2.weight" in rest:
                new_k = f"encoder.blocks.{idx}.norm2.weight"
            elif "norm2.bias" in rest:
                new_k = f"encoder.blocks.{idx}.norm2.bias"
            elif "mlp.fc1.weight" in rest:
                new_k = f"encoder.blocks.{idx}.mlp.fc1.weight"
            elif "mlp.fc1.bias" in rest:
                new_k = f"encoder.blocks.{idx}.mlp.fc1.bias"
            elif "mlp.fc2.weight" in rest:
                new_k = f"encoder.blocks.{idx}.mlp.fc2.weight"
            elif "mlp.fc2.bias" in rest:
                new_k = f"encoder.blocks.{idx}.mlp.fc2.bias"

        # Final encoder norm
        elif k == "norm.weight":
            new_k = "encoder.norm.weight"
        elif k == "norm.bias":
            new_k = "encoder.norm.bias"

        # Skip head (segmentation task)
        elif "head." in k:
            print(f"Skipping head: {k}")
            continue

        # Assign weight if exists in Prithvi's checkpoint
        if new_k in prithvi_sd:
            new_sd[k] = prithvi_sd[new_k]
        else:
            print(f"Skipping {k} → {new_k}: not found in Prithvi checkpoint.")

    # Load the weights into your model
    custom_model.load_state_dict(new_sd, strict=False)
    print("✅ Successfully loaded Prithvi encoder weights into custom model (head skipped).")


class SegmentationDecoder(nn.Module):
    def __init__(self, config, num_classes=2):
        super().__init__()
        grid_size = config.image_size // config.patch_size  # E.g., 16 if 224/14

        self.grid_size = grid_size
        self.hidden_dim = config.hidden_dim

        self.conv1 = nn.Conv2d(config.hidden_dim, config.hidden_dim // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(config.hidden_dim // 2)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(config.hidden_dim // 2, num_classes, kernel_size=1)

    def forward(self, x):
        # x: [B, N+1, D]
        B, N_plus_1, D = x.shape
        x = x[:, 1:, :]  # Remove cls token

        # Reshape to feature map
        x = x.transpose(1, 2).reshape(B, D, self.grid_size, self.grid_size)  # [B, D, H', W']

        # Decode
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)

        # Upsample to input resolution
        x = F.interpolate(x, scale_factor=(self.grid_size * config.patch_size) // self.grid_size, mode='bilinear', align_corners=False)
        return x  # [B, num_classes, H, W]


class UNetDecoder(nn.Module):
    def __init__(self, config, num_classes=2):
        super().__init__()
        self.grid_size = config.image_size // config.patch_size

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_dim, config.hidden_dim//2, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(config.hidden_dim//2, config.hidden_dim//2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_dim//2, config.hidden_dim//4, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(config.hidden_dim//4, config.hidden_dim//4, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.final_conv = nn.Conv2d(config.hidden_dim//4, num_classes, kernel_size=1)

    def forward(self, x):
        x = x[:, 1:, :]  # Remove cls token
        B, N, D = x.shape
        h = w = self.grid_size
        x = x.transpose(1, 2).reshape(B, D, h, w)

        x = self.up1(x)
        x = self.up2(x)

        out = self.final_conv(x)
        out = F.interpolate(out, size=(self.grid_size * config.patch_size, self.grid_size * config.patch_size), mode='bilinear', align_corners=False)
        return out


class PrithviDecoder(nn.Module):
    def __init__(self, config, num_classes=2):
        super().__init__()
        self.grid_size = config.image_size // config.patch_size
        hidden_dim = config.hidden_dim

        self.block1 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=2, stride=2)

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.GELU()
        )

        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.GELU()
        )

        self.block4 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim // 8, hidden_dim // 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim // 16),
            nn.GELU()
        )

        self.head = nn.Conv2d(hidden_dim // 16, num_classes, kernel_size=1)

    def forward(self, x):
        x = x[:, 1:, :]  # Remove cls token
        B, N, D = x.shape
        h = w = self.grid_size
        x = x.transpose(1, 2).reshape(B, D, h, w)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        out = self.head(x)
        out = F.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)
        return out


def evaluate_segmentation(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    miou_scores = []
    all_ious = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device).long()

            outputs = model(images)  # [B, C, H, W]
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)  # [B, H, W]
            miou, ious = compute_mIoU(preds.cpu(), masks.cpu())
            miou_scores.append(miou)
            all_ious.append(ious)

    avg_val_loss = val_loss / len(val_loader)
    avg_miou = np.nanmean(miou_scores)
    avg_ious = np.nanmean(np.array(all_ious), axis=0)

    print(f"Validation Loss: {avg_val_loss:.4f}, mIoU: {avg_miou:.4f}, IoUs: {np.round(avg_ious, 4)}")
    return avg_val_loss, avg_miou, avg_ious


def unfreeze_last_encoder_blocks(vit, n_last_blocks=4):
    """
    Unfreeze the last n transformer blocks of a VIT (your Prithvi encoder)
    and the final encoder norm.
    """
    total = vit.config.n_layer
    start = max(0, total - n_last_blocks)

    # Freeze everything first (defensive)
    for p in vit.parameters():
        p.requires_grad = False

    # Unfreeze last K blocks
    for name, p in vit.named_parameters():
        # Matches both "encoder.blocks_12.qkv.weight" and "blocks_12.qkv.weight"
        m = re.search(r'blocks_(\d+)\.', name)
        if m and int(m.group(1)) >= start:
            p.requires_grad = True

        # Final encoder norm (top-level vit.norm OR vit.encoder.norm)
        if name.startswith("norm.") or name.startswith("encoder.norm"):
            p.requires_grad = True


def print_trainable_summary(vit):
    total, trainable = 0, 0
    for p in vit.parameters():
        num = p.numel()
        total += num
        if p.requires_grad:
            trainable += num
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")


@torch.no_grad()
def estimate_pos_fraction(loader, ignore_index=255, max_batches=50):
    """Rough fraction of positive (class=1) pixels across a sample of the train loader."""
    pos, tot = 0, 0
    for i, (_, masks) in enumerate(loader):
        if i >= max_batches:
            break
        masks = masks.long()
        valid = (masks != ignore_index)
        pos += (masks == 1).masked_select(valid).numel()
        tot += valid.sum().item()
    return max(1e-6, pos / max(tot, 1))


class CombinedLoss(nn.Module):
    """Weighted CE + soft Dice on foreground (class=1)."""
    def __init__(self, ce_weights, dice_weight=0.5, ignore_index=255, smooth=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weights, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        # CE
        loss_ce = self.ce(logits, targets)

        # Dice on foreground channel
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # Build one-hot but zero-out ignored pixels
        t = targets.clone()
        t[t == self.ignore_index] = 0
        one_hot = F.one_hot(t, num_classes).permute(0, 3, 1, 2).float()
        valid = (targets != self.ignore_index).unsqueeze(1).float()

        p1 = probs[:, 1:2] * valid
        g1 = one_hot[:, 1:2] * valid

        inter = (p1 * g1).sum(dim=(0, 2, 3))
        denom = p1.sum(dim=(0, 2, 3)) + g1.sum(dim=(0, 2, 3)) + self.smooth
        dice = 1.0 - (2.0 * inter + self.smooth) / denom
        loss_dice = dice.mean()

        return loss_ce + self.dice_weight * loss_dice


# def train_segmentation(model, train_loader, val_loader, config, epochs=20, lr=1e-4, device="cuda"):
#     model = model.to(device)
#     criterion = nn.CrossEntropyLoss(ignore_index=255)
#     optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=lr)

#     train_losses = []
#     val_losses = []
#     best_miou = 0.0
#     best_model_path = "best_model.pth"

#     for epoch in range(epochs):
#         model.train()
#         running_train_loss = 0.0
#         loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

#         for images, masks in loop:
#             images = images.to(device)
#             masks = masks.to(device).long()

#             outputs = model(images)
#             loss = criterion(outputs, masks)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             running_train_loss += loss.item()
#             loop.set_postfix(loss=loss.item())

#         avg_train_loss = running_train_loss / len(train_loader)
#         train_losses.append(avg_train_loss)
#         print(f"\nTrain Loss: {avg_train_loss:.4f}")

#         # Validation
#         model.eval()
#         running_val_loss = 0.0
#         miou_scores = []
#         all_ious = []

#         with torch.no_grad():
#             for images, masks in val_loader:
#                 images = images.to(device)
#                 masks = masks.to(device).long()

#                 outputs = model(images)
#                 val_loss = criterion(outputs, masks)
#                 running_val_loss += val_loss.item()

#                 preds = torch.argmax(outputs, dim=1)
#                 miou, ious = compute_mIoU(preds.cpu(), masks.cpu())
#                 miou_scores.append(miou)
#                 all_ious.append(ious)

#         avg_val_loss = running_val_loss / len(val_loader)
#         avg_miou = np.nanmean(miou_scores)
#         avg_ious = np.nanmean(np.array(all_ious), axis=0)

#         val_losses.append(avg_val_loss)

#         print(f"Validation Loss: {avg_val_loss:.4f}, mIoU: {avg_miou:.4f}, IoUs: {np.round(avg_ious, 4)}")

#         # Save best model
#         if avg_miou > best_miou:
#             best_miou = avg_miou
#             torch.save(model.state_dict(), best_model_path)
#             print(f"✅ Best model saved with mIoU: {best_miou:.4f}")

#     print("Training complete.")
#     return train_losses, val_losses


def train_segmentation(model, train_loader, val_loader, config, epochs=150, device="cuda",
                       n_unfreeze_blocks=4, dice_weight=0.5, patience=15):
    model = model.to(device)

    # --- class weights from data ---
    pos_frac = estimate_pos_fraction(train_loader, ignore_index=255, max_batches=50)
    w_bg = 1.0
    w_pos = max(1.0, (1.0 - pos_frac) / pos_frac)  # inverse freq
    ce_weights = torch.tensor([w_bg, w_pos], dtype=torch.float32, device=device)

    criterion = CombinedLoss(ce_weights=ce_weights, dice_weight=dice_weight, ignore_index=255)

    # --- unfreeze last K encoder blocks ---
    unfreeze_last_encoder_blocks(model.encoder, n_last_blocks=n_unfreeze_blocks)
    print_trainable_summary(model.encoder)

    # --- optimizer with param groups ---
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    dec_params = list(model.decoder.parameters())

    optimizer = torch.optim.AdamW([
        {"params": dec_params, "lr": 5e-5},
        {"params": enc_params, "lr": 1e-5},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    train_losses, val_losses = [], []
    best_miou = 0.0
    wait = 0

    for epoch in range(epochs):
        model.train()
        running_train = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]"):
            images = images.to(device)
            masks = masks.to(device).long()

            logits = model(images)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec_params + enc_params, max_norm=1.0)
            optimizer.step()

            running_train += loss.item()

        avg_train = running_train / max(1, len(train_loader))
        train_losses.append(avg_train)
        print(f"\nTrain Loss: {avg_train:.4f}")

        # --- validation ---
        model.eval()
        running_val = 0.0
        miou_scores, all_ious = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device).long()
                logits = model(images)
                val_loss = criterion(logits, masks)
                running_val += val_loss.item()

                preds = torch.argmax(logits, dim=1)
                miou, ious = compute_mIoU(preds.cpu(), masks.cpu())
                miou_scores.append(miou)
                all_ious.append(ious)

        avg_val = running_val / max(1, len(val_loader))
        avg_miou = np.nanmean(miou_scores)
        avg_ious = np.nanmean(np.array(all_ious), axis=0)

        val_losses.append(avg_val)
        print(f"Validation Loss: {avg_val:.4f}, mIoU: {avg_miou:.4f}, IoUs: {np.round(avg_ious, 4)}")

        # step scheduler and early stop
        scheduler.step(avg_val)

        improved = avg_miou > best_miou + 0.002  # ~0.2% abs improvement
        if improved:
            best_miou = avg_miou
            wait = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ Best model saved with mIoU: {best_miou:.4f}")
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1} (best mIoU={best_miou:.4f})")
                break

    print("Training complete.")
    return train_losses, val_losses


class VineyardSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_list, target_size=(224, 224), transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.target_size = target_size

        self.image_files = sorted(file_list)
        self.selected_bands = [2, 3, 4, 8, 11, 12]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_filename)

        mask_filename = f"mask_{img_filename.replace('.tif', '_ALL.png')}"
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # Load Sentinel-2 selected bands
        with rasterio.open(img_path) as src:
            image = src.read(self.selected_bands).astype(np.float32)  # [6, H, W]

        image = image / 10000.0

        # Convert to torch tensor & resize
        image = torch.from_numpy(image)  # [6, H, W]
        image = TF.resize(image, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)  # Resize image

        # Add time dimension: [6, 1, H, W]
        image = image.unsqueeze(1)

        # Load mask
        mask = Image.open(mask_path)
        mask = np.array(mask).astype(np.int64)  # [H, W]
        mask = torch.from_numpy(mask)

        mask = TF.resize(mask.unsqueeze(0), self.target_size, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)  # Use NEAREST for masks

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask


class Sen2HandFloodDataset(Dataset):
    def __init__(self, s2_dir, label_dir, file_list, target_size=(224, 224), normalize=True):
        self.s2_dir = s2_dir
        self.label_dir = label_dir
        self.file_list = file_list
        self.target_size = target_size
        self.normalize = normalize

        # Use 6 informative Sentinel-2 bands (Blue, Green, Red, NIR, SWIR1, SWIR2)
        self.selected_bands = [1, 2, 3, 7, 11, 12]  # 0-based indices

        self.mean = torch.tensor([
            954.04,   # Band 2 (Blue)
            936.18,   # Band 3 (Green)
            614.69,   # Band 4 (Red)
            2787.74,  # Band 8 (NIR)
            1741.34,  # Band 12 (SWIR1)
            720.27    # Band 13 (SWIR2)
        ], dtype=torch.float32)

        self.std = torch.tensor([
            206.00,
            221.87,
            281.75,
            729.40,
            528.61,
            312.91
        ], dtype=torch.float32)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base_name = self.file_list[idx]  # e.g., 'Bolivia_23014'

        s2_path = os.path.join(self.s2_dir, f"{base_name}_S2Hand.tif")
        label_path = os.path.join(self.label_dir, f"{base_name}_LabelHand.tif")

        with rasterio.open(s2_path) as src:
            img = src.read(indexes=[i + 1 for i in self.selected_bands]).astype(np.float32)  # [6, H, W]

        img = torch.from_numpy(img)

        if self.normalize:
            img = (img - self.mean[:, None, None]) / self.std[:, None, None]

        img = TF.resize(img, self.target_size, interpolation=TF.InterpolationMode.BILINEAR)
        img = img.unsqueeze(1)  # [6, 1, H, W]

        with rasterio.open(label_path) as src:
            mask = src.read(1).astype(np.int64)

        mask = torch.from_numpy(mask)

        # Map -1 → 255 (ignored class)
        mask[mask == -1] = 255

        # Resize after remapping
        mask = TF.resize(mask.unsqueeze(0), self.target_size, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

        return img, mask





class VineyardSegmentationModel(nn.Module):
    def __init__(self, encoder, config, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.decoder = PrithviDecoder(config, num_classes)

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        # with torch.no_grad():
        features = self.encoder(x, return_features=True)  # ✅ Forward through frozen encoder

        out = self.decoder(features)
        return out


def compute_mIoU(preds, labels, num_classes=2, ignore_index=255):
    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = (preds == cls)
        label_inds = (labels == cls)

        intersection = (pred_inds & label_inds).sum().item()
        union = (pred_inds | label_inds).sum().item()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)

    return np.nanmean(ious), ious


def visualize_prediction(model, val_loader, device="cuda", save_path="prediction_example.png"):
    model.eval()
    model.to(device)

    images, masks = next(iter(val_loader))
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)  # [B, num_classes, H, W]
        preds = torch.argmax(outputs, dim=1)

    idx = np.random.randint(0, images.size(0))

    image = images[idx].cpu().numpy()  # [6, 1, H, W]
    mask = masks[idx].cpu().numpy()
    pred = preds[idx].cpu().numpy()

    image = image[:, 0, :, :]  # Remove time dim → [6, H, W]

    # Use bands 3, 2, 1 (Green, Red, NIR) for RGB-like visual (adjust if needed)
    nir = image[3]
    red = image[2]
    green = image[1]

    rgb = np.stack([nir, red, green], axis=-1)
    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))

    # Plot
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    plt.title("Input Image (NIR-R-G)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    # Save to file
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Prediction example saved to: {save_path}")




if __name__ == '__main__':
    pkl_path = 'Prithvi_EO_V2_600M.pt'
    state_dict = torch.load(pkl_path, map_location='cpu')
    state_dict = {k: v for k, v in state_dict.items() if k.startswith('encoder')}

    import os
    from sklearn.model_selection import train_test_split

    image_dir = '/home/ubuntu/prithvi/v1.1/data/flood_events/HandLabeled/S2Hand'
    mask_dir = '/home/ubuntu/prithvi/v1.1/data/flood_events/HandLabeled/LabelHand'

    # Get base names only (without _S2Hand.tif)
    all_image_files = sorted([
        f.replace('_S2Hand.tif', '') for f in os.listdir(image_dir) if f.endswith('_S2Hand.tif')
    ])

    # Split into train/val
    train_files, val_files = train_test_split(all_image_files, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = Sen2HandFloodDataset(image_dir, mask_dir, train_files, target_size=(224, 224))
    val_dataset = Sen2HandFloodDataset(image_dir, mask_dir, val_files, target_size=(224, 224))

    print(train_dataset[0][0].shape)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)


    vit_encoder = VIT(config)  # Your Prithvi model with encoder weights loaded
    load_prithvi_encoder(vit_encoder, state_dict)

    model = VineyardSegmentationModel(encoder=vit_encoder, config=config, num_classes=2)

    train_losses, val_losses = train_segmentation(model, train_loader, val_loader, config,
    epochs=150, device="cuda", n_unfreeze_blocks=4, dice_weight=0.5, patience=15
    )

    visualize_prediction(model, val_loader, device="cuda", save_path="senflood_prediction.png")

    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')  # Save the plot as an image file
    plt.close()  # Close the plot to avoid display in notebooks or scripts
