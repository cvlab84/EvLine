import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
import pandas as pd
import numpy as np
import csv


def build_hough_bank(image_height, image_width, theta_res=3.0, rho_res=1.0):
    """
    Generate a Hough bank tensor for all pixels in an image.

    Args:
        image_height (int): Height of the image (rows)
        image_width (int): Width of the image (cols)
        theta_res (float): Resolution of theta in degrees
        rho_res (float): Resolution of rho in pixels

    Returns:
        bank: A tensor of shape (H, W, R, T) where
              R = number of rho bins
              T = number of theta bins
              Each (y,x) has a 2D binary mask over (rho, theta)
    """
    # 1. Define theta values
    theta = np.arange(0, 180, theta_res)  # [0, 180)
    T = len(theta)
    cos_t = np.cos(np.deg2rad(theta))
    sin_t = np.sin(np.deg2rad(theta))

    # 2. Calculate HALF maximum rho (논문 스타일: [-L/2, +L/2])
    diag_len = np.hypot(image_height, image_width)
    rho_max = diag_len / 2
    R = int(np.ceil(diag_len / rho_res))  # 논문: R = diag_len / Δr
    rho = np.linspace(-rho_max, rho_max, R)

    # 3. Create coordinate grid and center it
    y_coords, x_coords = np.meshgrid(np.arange(image_height), np.arange(image_width), indexing='ij')
    y_centered = y_coords - image_height / 2
    x_centered = x_coords - image_width / 2
    coords = np.stack([x_centered, y_centered], axis=-1)  # (H, W, 2)

    # 4. Compute rho values for each pixel and theta
    bank = np.zeros((image_height, image_width, R, T), dtype=np.uint8)
    for t in range(T):
        rho_vals = coords[..., 0] * cos_t[t] + coords[..., 1] * sin_t[t]  # shape: (H, W)
        rho_indices = np.round((rho_vals + rho_max) / rho_res).astype(np.int32)  # shift to [0, R)

        valid = (rho_indices >= 0) & (rho_indices < R)
        y_idx, x_idx = np.nonzero(valid)
        r_idx = rho_indices[y_idx, x_idx]

        bank[y_idx, x_idx, r_idx, t] = 1

    return bank, rho, theta, coords


# ───────────────────────────────
# ✅ Patch ID map 생성 함수
def make_patch_id_map(H, W, patch, device):
    n_h, n_w = H // patch, W // patch
    y = torch.arange(H, device=device) // patch
    x = torch.arange(W, device=device) // patch
    return (y.view(H, 1) * n_w + x.view(1, W)).long()

def positional_encoding(num_patches, dim, device=None):
    """
    Sinusoidal positional encoding for patches
    num_patches: number of patches
    dim: embedding dimension
    """
    position = torch.arange(num_patches, dtype=torch.float, device=device).unsqueeze(1)  # (num_patches, 1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * -(math.log(10000.0) / dim))
    pe = torch.zeros(num_patches, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (num_patches, dim)

def Event_embedd(x, patch_size, device):
    """
    x: (B, 1, H, W) - single channel, single batch
    return: (num_patches, patch*patch)
    """
    B, C, H, W = x.shape
    assert C == 1
    x_unfold = torch.nn.functional.unfold(x, kernel_size=patch_size, stride=patch_size)  # (1, patch*patch, num_patches)
    patch_embeddings = x_unfold.transpose(1, 2)  # (B, num_patches, patch*patch)

    # ✅ 포지셔널 인코딩 생성
    num_patches = patch_embeddings.size(1)
    dim = patch_embeddings.size(2)
    pe = positional_encoding(num_patches, dim, device=device)  # (num_patches, dim)

    # ✅ 더하기 (batch마다 동일하게 broadcast)
    patch_embeddings = patch_embeddings + pe.unsqueeze(0)

    return patch_embeddings
# ───────────────────────────────
# ✅ 패치 기준 Hough 맵 합산 함수
def Hough_embedd(feat, patch_id_map, num_patches):
    # feat: (H,W,r,t) or (B,H,W,r,t)
    if feat.dim() == 4:
        H, W, r, t = feat.shape
        feat_flat = feat.view(-1, r, t)      # (H*W,r,t)
        patch_ids = patch_id_map.view(-1)    # (H*W,)
        out = torch.zeros(num_patches, r, t, device=feat.device)
        out.index_add_(0, patch_ids, feat_flat)
        return out
    elif feat.dim() == 5:
        B, H, W, r, t = feat.shape
        patch_ids = patch_id_map.view(-1)    # (H*W,)
        out = torch.zeros(B, num_patches, r, t, device=feat.device)
        feat_flat = feat.view(B, H*W, r, t)
        for b in range(B):
            out[b].index_add_(0, patch_ids, feat_flat[b])
        return out
    else:
        raise ValueError("feat must be 4D or 5D")

# ▍포컬 로스 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)  # 예측이 맞을수록 pt ↑
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

def rescale_and_change_xytoyx_gt_tensor(gt_tensor, orig_w, orig_h, new_w, new_h):
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    resized = gt_tensor.clone()
    x_scaled = resized[..., 0] * scale_x  # 원래 x
    y_scaled = resized[..., 1] * scale_y  # 원래 y

    # 🔄 (x,y) → (y,x) 좌표계 변환
    resized[..., 0] = y_scaled
    resized[..., 1] = x_scaled
    return resized

def visualize_hough_embedding(hough_embed, num_images=3):
    """
    hough_embed: (B, num_patches, R, T)
    num_images: 몇 개 패치 시각화할지
    """
    B, num_patches, R, T = hough_embed.shape

    # 첫 번째 배치만 확인
    hough_sample = hough_embed[0].detach().cpu()  # (num_patches, R, T)

    # 패치 중 일부 랜덤 선택
    idxs = torch.linspace(0, num_patches - 1, steps=num_images).long()

    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    if num_images == 1:
        axes = [axes]
    for ax, idx in zip(axes, idxs):
        img = hough_sample[idx]
        ax.imshow(img, cmap='jet')
        ax.set_title(f"Patch {idx.item()}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

class MaskedFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        """
        inputs: (B, ..., H, W) 예측값
        targets: 동일 shape GT
        mask: (B, ..., H, W) bool 텐서. True=유효, False=패딩
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if mask is not None:
            focal_loss = focal_loss * mask  # 마스크 적용

        if self.reduction == "mean":
            denom = mask.sum() if mask is not None else focal_loss.numel()
            return focal_loss.sum() / (denom + 1e-6)
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

def Event_reconstruct(patch_embeddings, patch_size, H, W):
    """
    patch_embeddings: (B, num_patches, patch*patch)
    return: (B, 1, H, W)
    """
    B, num_patches, patch_dim = patch_embeddings.shape
    # (B, patch*patch, num_patches) 로 다시 변환
    x_unfold = patch_embeddings.transpose(1, 2)
    # fold로 원본 복원
    x_recon = F.fold(x_unfold, output_size=(H, W), kernel_size=patch_size, stride=patch_size)
    return x_recon

def visualize_hough_comparison(hough_embed, patch_size, num_patches_to_show=4, random_pick=True):
    """
    hough_embed: (B, num_patches, R, T)
    patch_size: 패치의 H*W 값 (정규화 기준)
    num_patches_to_show: 표시할 패치 개수
    random_pick: 랜덤으로 패치 선택 여부
    """
    B, N, R, T = hough_embed.shape

    # 🔹 Global normalization
    global_max = hough_embed.amax(dim=(1, 2, 3), keepdim=True)
    hough_global = hough_embed / (global_max + 1e-6)

    # 🔹 Patch-wise normalization
    patch_max = hough_embed.amax(dim=(2, 3), keepdim=True)
    hough_patch = hough_embed / (patch_max + 1e-6)

    # 🔹 sqrt(patch_size) normalization
    sqrt_norm = hough_embed / (patch_size**0.5)

    # 🔹 첫 번째 배치만 보기
    b = 0

    # 🔹 패치 선택
    if random_pick:
        patch_indices = random.sample(range(N), min(num_patches_to_show, N))
    else:
        patch_indices = torch.linspace(0, N-1, num_patches_to_show, dtype=torch.long).tolist()

    fig, axes = plt.subplots(len(patch_indices), 4, figsize=(12, 3 * len(patch_indices)))
    fig.suptitle(f"Hough Normalization Comparison (Batch {b})", fontsize=14)

    vmax_global = hough_embed.max().item()  # 전역 최대값으로 고정

    for row, idx in enumerate(patch_indices):
        idx = int(idx)

        # Original
        axes[row, 0].imshow(hough_embed[b, idx].cpu(), cmap='jet', vmin=0, vmax=vmax_global)
        axes[row, 0].set_title(f"Original {idx}")
        axes[row, 0].axis("off")

        # Global Norm
        axes[row, 1].imshow(hough_global[b, idx].cpu(), cmap='jet', vmin=0, vmax=1)
        axes[row, 1].set_title("Global Norm")
        axes[row, 1].axis("off")

        # Patch-wise Norm
        axes[row, 2].imshow(hough_patch[b, idx].cpu(), cmap='jet', vmin=0, vmax=1)
        axes[row, 2].set_title("Patch-wise Norm")
        axes[row, 2].axis("off")

        # sqrt(patch_size) Norm
        axes[row, 3].imshow(sqrt_norm[b, idx].cpu(), cmap='jet', vmin=0, vmax=vmax_global/(patch_size**0.5))
        axes[row, 3].set_title("Sqrt(PatchSize) Norm")
        axes[row, 3].axis("off")

    plt.tight_layout()
    plt.show()

    def compare_norm_stats(hough_embed, patch_size):
        """
        hough_embed: (B, num_patches, R, T)
        patch_size: 패치 픽셀 개수(H*W)
        """
        # 🔹 Global normalization
        global_max = hough_embed.amax(dim=(1, 2, 3), keepdim=True)
        hough_global = hough_embed / (global_max + 1e-6)

        # 🔹 Patch-wise normalization
        patch_max = hough_embed.amax(dim=(2, 3), keepdim=True)
        hough_patch = hough_embed / (patch_max + 1e-6)

        # 🔹 sqrt(patch_size) normalization
        sqrt_norm = hough_embed / (patch_size ** 0.5)

        # 🔹 패치별 최대값
        max_global = hough_global.amax(dim=(2, 3))  # (B, N)
        max_patch = hough_patch.amax(dim=(2, 3))  # (B, N)
        max_sqrt = sqrt_norm.amax(dim=(2, 3))  # (B, N)

        # 🔹 Global vs Patch L1/L2 차이
        diff_l1_gp = torch.abs(hough_global - hough_patch).mean().item()
        diff_l2_gp = torch.sqrt(((hough_global - hough_patch) ** 2).mean()).item()

        # 🔹 Global vs Sqrt L1/L2 차이
        diff_l1_gs = torch.abs(hough_global - sqrt_norm).mean().item()
        diff_l2_gs = torch.sqrt(((hough_global - sqrt_norm) ** 2).mean()).item()

        # 🔹 Patch vs Sqrt L1/L2 차이
        diff_l1_ps = torch.abs(hough_patch - sqrt_norm).mean().item()
        diff_l2_ps = torch.sqrt(((hough_patch - sqrt_norm) ** 2).mean()).item()

        # 🔹 출력
        print("=== 패치별 최대값 ===")
        print(f"Global Norm 평균: {max_global.mean().item():.4f}, 표준편차: {max_global.std().item():.4f}")
        print(f"Patch-wise Norm 평균: {max_patch.mean().item():.4f}, 표준편차: {max_patch.std().item():.4f}")
        print(f"Sqrt(PatchSize) Norm 평균: {max_sqrt.mean().item():.4f}, 표준편차: {max_sqrt.std().item():.4f}")

        print("\n=== 정규화 방식 간 L1/L2 차이 ===")
        print(f"Global vs Patch: L1={diff_l1_gp:.6f}, L2={diff_l2_gp:.6f}")
        print(f"Global vs Sqrt : L1={diff_l1_gs:.6f}, L2={diff_l2_gs:.6f}")
        print(f"Patch vs Sqrt  : L1={diff_l1_ps:.6f}, L2={diff_l2_ps:.6f}")

        return max_global, max_patch, max_sqrt

# 🔥 collate_fn 추가: shape이 다른 텐서를 리스트로 묶기
def pad_and_mask(batch):
    imgs, labels = zip(*batch)  # imgs: (C,H,W), labels: (N,2,2)
    imgs = torch.stack(imgs)  # 입력 텐서는 바로 배치화

    max_lines = max([lbl.shape[0] for lbl in labels])  # 배치 내 최대 라인 수
    padded_labels = []
    masks = []
    for lbl in labels:
        n = lbl.shape[0]
        # padding
        pad = torch.zeros((max_lines, 2, 2), dtype=lbl.dtype)
        pad[:n] = lbl
        padded_labels.append(pad)
        # mask
        m = torch.zeros((max_lines,), dtype=torch.bool)
        m[:n] = 1
        masks.append(m)
    padded_labels = torch.stack(padded_labels)  # (B, max_lines, 2, 2)
    masks = torch.stack(masks)  # (B, max_lines)
    return imgs, padded_labels, masks

def load_checkpoint(model, optimizer, scheduler, path, device):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)

        # --- 모델 ---
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)  # ← 반드시 필요!!!
        else:
            print("⚠️ model_state_dict 없음 → 모델은 초기 상태로 유지")

        # --- 옵티마이저 ---
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("⚠️ optimizer_state_dict 없음 → optimizer 초기 상태 유지")

        # --- 스케줄러 ---
        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f"⚠️ scheduler_state_dict 로드 실패 → 무시 ({e})")
        else:
            print("⚠️ scheduler_state_dict 없음 → scheduler 초기 상태 유지")

        # --- 기타 정보 ---
        start_epoch = checkpoint.get('epoch', 0)
        best_loss   = checkpoint.get('loss', float('inf'))
        best_acc    = checkpoint.get('acc', 0.0)

        print(f"🔄 체크포인트 로드 완료: {path} "
              f"(epoch={start_epoch}, best_loss={best_loss:.4f}, best_acc={best_acc:.4f})")

        return model, optimizer, scheduler, start_epoch, best_loss, best_acc

    else:
        print("⚠️ 체크포인트 없음 → 새 학습 시작")
        return model, optimizer, scheduler, 0, float('inf'), 0.0

def save_checkpoint(epoch, model, optimizer, scheduler, loss, acc, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'acc':  acc
    }
    torch.save(checkpoint, path)
    print(f"✅ 체크포인트 저장: {path}")

def masked_mse_loss(pred, target, mask=None, reduction="mean"):
    """
    pred, target: (B, ..., H, W)
    mask: (B, ..., H, W), 0=무시, 1=유효
    """
    loss = F.mse_loss(pred, target, reduction="none")
    if mask is not None:
        loss = loss * mask
    if reduction == "mean":
        denom = mask.sum() if mask is not None else loss.numel()
        return loss.sum() / (denom + 1e-6)
    elif reduction == "sum":
        return loss.sum()
    return loss

def hough_embedd_chunk(X, patch_id_map, hbank, num_patches, chunk=1000):
    """
    메모리 절약 버전 (Chunk 단위로 곱셈) - 올바른 버전
    """
    B, _, H, W = X.shape
    R, T = hbank.shape[-2:]
    out = torch.zeros(B, num_patches, R, T, device=X.device)
    patch_ids = patch_id_map.view(-1)
    hbank_flat = hbank.view(H*W, R*T)

    for b in range(B):
        X_flat = X[b].view(-1)  # (H*W,)
        out_b = torch.zeros(num_patches, R, T, device=X.device)  # ✅ 여기로 이동
        for start in range(0, H*W, chunk):
            end = min(start + chunk, H*W)
            contrib = X_flat[start:end].unsqueeze(1) * hbank_flat[start:end]  # (chunk, R*T)
            contrib = contrib.view(-1, R, T)
            out_b.index_add_(0, patch_ids[start:end], contrib)
        out[b] = out_b
    return out

import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_heatmap_comparison(
    heatmap, heatmap_gt, save_dir, epoch, step, sample_count, folder, fname, interval=500
):
    """
    예측 히트맵과 GT 히트맵을 주기적으로 저장하는 함수

    Args:
        heatmap (torch.Tensor): 모델이 출력한 heatmap 텐서, shape (B, 1, H, W) 예상
        heatmap_gt (torch.Tensor): GT heatmap 텐서, shape (B, 1, H, W) 예상
        save_dir (str): 저장 경로
        epoch (int): 현재 epoch
        step (int): 현재 step
        sample_count (int): 저장 주기를 제어하기 위한 샘플 카운트
        interval (int): 저장할 간격 (기본값 500)
    """
    if sample_count % interval != 0:
        return  # 지정한 간격이 아닐 때는 그냥 리턴

    os.makedirs(save_dir, exist_ok=True)

    # 🔹 heatmap 시각화
    pred_img = heatmap[0].detach().cpu().squeeze()  # (H, W)
    gt_img = heatmap_gt[0].detach().cpu().squeeze()  # (H, W)

    plt.figure(figsize=(8, 4))
    plt.suptitle(f"{folder[0]}/{fname[0]}", fontsize=10)

    plt.subplot(1, 2, 1)
    plt.imshow(pred_img, cmap='jet')
    plt.title("Predicted Heatmap")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(gt_img, cmap='jet')
    plt.title("GT Heatmap")
    plt.axis("off")


    plt.tight_layout()
    # 🔹 파일 이름에 폴더명/파일명 추가
    safe_folder = os.path.basename(folder[0])  # 혹시 경로 전체가 들어있을 경우 대비
    safe_fname = os.path.splitext(fname[0])[0] # 확장자 제거
    save_path = os.path.join(
        save_dir, f"epoch{epoch}_step{step}_{safe_folder}_{safe_fname}.png"
    )

    plt.savefig(save_path, dpi=150)
    plt.close()

def save_val_heatmap_comparison(
    heatmap, heatmap_gt, save_dir, epoch, folder, fname, seen_folders
):
    """
    Validation 단계에서 시나리오별로 하나씩만 시각화 저장
    에포크별 하위 폴더에 저장되도록 수정
    """
    # ✅ 에포크별 하위 폴더 생성
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    for b in range(len(folder)):
        scenario = os.path.basename(folder[b])
        if scenario in seen_folders:
            continue  # 이미 저장했으면 skip

        # 🔹 heatmap 시각화
        pred_img = heatmap[b].detach().cpu().squeeze()
        gt_img = heatmap_gt[b].detach().cpu().squeeze()

        plt.figure(figsize=(8, 4))
        plt.suptitle(f"{scenario}/{fname[b]}", fontsize=10)

        plt.subplot(1, 2, 1)
        plt.imshow(pred_img, cmap='jet')
        plt.title("Predicted Heatmap")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(gt_img, cmap='jet')
        plt.title("GT Heatmap")
        plt.axis("off")

        plt.tight_layout()

        # ✅ 파일 경로: epoch 폴더 안에 저장
        save_path = os.path.join(
            epoch_dir, f"{scenario}_{os.path.splitext(fname[b])[0]}.png"
        )
        plt.savefig(save_path, dpi=150)
        plt.close()

        seen_folders.add(scenario)  # 저장 완료 표시


def plot_loss_curve(log_path, save_path=None, show=True):
    """
    CSV 로그 파일에서 train/val loss를 읽어와 그래프를 그려줍니다.

    Args:
        log_path (str | Path): loss_log.csv 경로
        save_path (str | Path, optional): 그래프를 저장할 경로 (None이면 저장하지 않음)
        show (bool): True면 plt.show() 실행
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"로그 파일을 찾을 수 없습니다: {log_path}")

    df = pd.read_csv(log_path)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
def reverse_mapping(point_list, numAngle, numRho, size=(32, 32)):
    #return type: [(y1, x1, y2, x2)]
    H, W = size
    irho = int(np.sqrt(H*H + W*W) + 1) / ((numRho - 1))
    itheta = np.pi / numAngle
    b_points = []

    for (ri, thetai) in point_list:
        theta = thetai * itheta
        r = ri - numRho // 2
        cosi = np.cos(theta) / irho
        sini = np.sin(theta) / irho
        if sini == 0:
            x = np.round(r / cosi + W / 2)
            b_points.append((0, int(x), H-1, int(x)))
        else:
            # print('k = %.4f', - cosi / sini)
            # print('b = %.2f', np.round(r / sini + W * cosi / sini / 2 + H / 2))
            angle = np.arctan(- cosi / sini)
            y = np.round(r / sini + W * cosi / sini / 2 + H / 2)
            p1, p2 = get_boundary_point(int(y), 0, angle, H, W)
            if p1 is not None and p2 is not None:
                b_points.append((p1[1], p1[0], p2[1], p2[0]))
    return b_points



def reverse_mapping_2(point_list, numRho=112,
                    theta_res=1.8, rho_res=math.sqrt(2)*2,
                    size=(224, 224)):
    """
    허프 공간 좌표 (theta index, rho index) -> 이벤트 맵 (H,W) 좌표계 직선 끝점
    반환: [(y1, x1, y2, x2)]
    """
    H, W = size
    itheta = np.deg2rad(theta_res)  # deg → rad 변환
    b_points = []

    for (ri, thetai) in point_list:
        theta = thetai * itheta
        r = (ri - numRho // 2) * rho_res  # 중심 기준 거리

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        if abs(sin_t) < 1e-6:
            # 수직선 (x = const)
            x = int(np.round(r / cos_t + W / 2))
            p1, p2 = (0, x), (H - 1, x)
        else:
            # 기울어진 직선
            angle = np.arctan(-cos_t / sin_t)
            y0 = np.round(r / sin_t + (W * cos_t) / (2 * sin_t) + H / 2)
            p1, p2 = get_boundary_point(int(y0), 0, angle, H, W)

        if p1 is not None and p2 is not None:
            b_points.append((p1[1], p1[0], p2[1], p2[0]))

    return b_points


def get_boundary_point(y, x, angle, H, W):
    '''
    Given point y,x with angle, return a two point in image boundary with shape [H, W]
    return point:[x, y]
    '''
    point1 = None
    point2 = None

    if angle == -np.pi / 2:
        point1 = (x, 0)
        point2 = (x, H - 1)
    elif angle == 0.0:
        point1 = (0, y)
        point2 = (W - 1, y)
    else:
        k = np.tan(angle)
        if y - k * x >= 0 and y - k * x < H:  # left
            if point1 == None:
                point1 = (0, int(y - k * x))
            elif point2 == None:
                point2 = (0, int(y - k * x))
                if point2 == point1: point2 = None
        # print(point1, point2)
        if k * (W - 1) + y - k * x >= 0 and k * (W - 1) + y - k * x < H:  # right
            if point1 == None:
                point1 = (W - 1, int(k * (W - 1) + y - k * x))
            elif point2 == None:
                point2 = (W - 1, int(k * (W - 1) + y - k * x))
                if point2 == point1: point2 = None
        # print(point1, point2)
        if x - y / k >= 0 and x - y / k < W:  # top
            if point1 == None:
                point1 = (int(x - y / k), 0)
            elif point2 == None:
                point2 = (int(x - y / k), 0)
                if point2 == point1: point2 = None
        # print(point1, point2)
        if x - y / k + (H - 1) / k >= 0 and x - y / k + (H - 1) / k < W:  # bottom
            if point1 == None:
                point1 = (int(x - y / k + (H - 1) / k), H - 1)
            elif point2 == None:
                point2 = (int(x - y / k + (H - 1) / k), H - 1)
                if point2 == point1: point2 = None
        # print(int(x-y/k+(H-1)/k), H-1)
        if point2 == None: point2 = point1
    return point1, point2

def line_to_boundary_intercepts(y0, x0, y1, x1, H=224, W=224):
    results = []
    dx = x1 - x0
    dy = y1 - y0
    # y=0
    if dy != 0:
        x = x0 + (0 - y0) * dx / dy
        if 0 <= x <= W-1:
            results.append((0, x))
    # y=H-1
    if dy != 0:
        x = x0 + ((H-1) - y0) * dx / dy
        if 0 <= x <= W-1:
            results.append((H-1, x))
    # x=0
    if dx != 0:
        y = y0 + (0 - x0) * dy / dx
        if 0 <= y <= H-1:
            results.append((y, 0))
    # x=W-1
    if dx != 0:
        y = y0 + ((W-1) - x0) * dy / dx
        if 0 <= y <= H-1:
            results.append((y, W-1))
    if len(results) >= 2:
        (y0n, x0n), (y1n, x1n) = results[:2]
        return [y0n, x0n, y1n, x1n]
    return None

# ───────────────────────────────
# ✅ Main
if __name__ == "__main__":
    # ▍Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ▍파라미터
    B, C, H, W = 1, 1, 224, 224
    patch = 16
    n_h, n_w = H // patch, W // patch
    num_patches = n_h * n_w
    rho, theta = 224, 180

    # ▍Patch ID map 생성
    patch_id_map = make_patch_id_map(H, W, patch, device)  # (H, W)
    pid = patch_id_map.view(-1)                            # (H*W,)

    # ▍Event 특징맵 생성
    #X = torch.randn(1, 1, 224, 320).cuda()
    X= patch_id_map.view(1,1,H,W).float()
    patch_embeddings = Event_embedd(X, patch)
    print("Event embedd ",patch_embeddings.shape)  # → (70, 1024)

    # ▍Hough 특징맵 생성
    hough = torch.randn(H, W, rho, theta, device=device)
    hough = X.view(H,W,1,1) * hough  # weihgt
    hough = Hough_embedd(hough, patch_id_map, num_patches)
    print("Hough merged shape:", hough.shape)  # (70, 18, 10)


    # ▍시각화
    plt.figure()
    plt.imshow(patch_id_map.cpu())
    plt.title(f"Patch ID map (H={H}, W={W}, patch={patch})")
    plt.colorbar()
    plt.tight_layout(); plt.show()

    plt.figure()
    plt.imshow(patch_embeddings.cpu())
    plt.title("Patch-wise Flattened Vectors (sorted)")
    plt.colorbar()
    plt.tight_layout(); plt.show()

    ##################################################################################
    H, W, rho, theta = 4, 4, 3, 3  # rho,theta 크게 설정
    num_patches = (H // 2) * (W // 2)  # 2x2 patch → 4 patches

    # 허프맵: 전부 1
    hough = torch.ones(H, W, rho, theta)

    # 픽셀 가중치 X
    X = torch.tensor([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [9., 10., 11., 12.],
        [13., 14., 15., 16.]
    ])

    # 가중치 반영
    hough = X.view(H, W, 1, 1) * hough  # (H,W,rho,theta)

    # patch ID map
    patch_id_map = torch.tensor([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3]
    ])


    def Hough_embedd(feat, patch_id_map, num_patches):
        H, W, r, t = feat.shape
        feat_flat = feat.view(-1, r, t)  # (H*W, r, t)
        patch_ids = patch_id_map.view(-1)  # (H*W,)
        out = torch.zeros(num_patches, r, t)
        out.index_add_(0, patch_ids, feat_flat)
        return out


    # 실행
    out = Hough_embedd(hough, patch_id_map, num_patches)
    print("out.shape:", out.shape)  # (4,3,3)
    print("패치0 결과:\n", out[0])
    print("패치1 결과:\n", out[1])
    print("패치2 결과:\n", out[2])
    ###################################################################################

import os
import cv2
import numpy as np

import os
import cv2
import numpy as np

import os
import cv2
import numpy as np

import cv2
import numpy as np
import os
import torch


def process_eval_step_RGB(save_root, data_root, folder_name, file_name, model_alias,
                      img_tensor, pred_coords, gt_lines,
                      hiou_func, tp_fp_fn_func, H, W,
                      stats_accumulator):
    """
    [img_tensor 직접 시각화 버전]
    1. 입력받은 img_tensor를 numpy/BGR 이미지로 변환
    2. 예측 좌표를 이미지 크기에 맞춰 스케일링 (이미지가 이미 모델 입력 크기면 scale=1)
    3. img_tensor 배경 위에 라인 드로잉 및 저장
    """

    # -------------------------------
    # 1. img_tensor를 이미지(BGR)로 변환
    # -------------------------------
    # 텐서가 (C, H, W) 형태라면 (H, W, C)로 변경
    if img_tensor.dim() == 3:
        # 대부분 (1, H, W) 또는 (3, H, W) 형태입니다.
        img_np = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    else:
        img_np = img_tensor.detach().cpu().numpy()

    # 정규화 해제 (0~1 -> 0~255)
    if img_np.max() <= 1.01:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)

    # 그레이스케일인 경우 컬러로 변환 (초록색 선을 그리기 위함)
    if img_np.shape[-1] == 1 or len(img_np.shape) == 2:
        img_final = cv2.cvtColor(img_np.squeeze(), cv2.COLOR_GRAY2BGR)
    else:
        img_final = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # -------------------------------
    # 2. 좌표 스케일링 설정
    # -------------------------------
    # img_tensor의 현재 크기
    curr_h, curr_w = img_final.shape[:2]

    # 모델 출력 좌표가 (H, W) 기준이므로, 현재 이미지 크기에 맞게 배율 조정
    scale_x = curr_w / W
    scale_y = curr_h / H

    # -------------------------------
    # 3. 예측 라인 그리기
    # -------------------------------
    if pred_coords is not None:
        for line in pred_coords:
            try:
                py1, px1, py2, px2 = line

                # 좌표 스케일링
                pt1 = (int(px1 * scale_x), int(py1 * scale_y))
                pt2 = (int(px2 * scale_x), int(py2 * scale_y))

                # 초록색 선 (B=0, G=255, R=0), 두께 2
                cv2.line(img_final, pt1, pt2, (0, 255, 0), 2)
            except Exception:
                continue

    # -------------------------------
    # 4. 저장 처리
    # -------------------------------
    vis_save_dir = os.path.join(save_root, str(folder_name), str(model_alias))
    os.makedirs(vis_save_dir, exist_ok=True)

    name_stem = os.path.splitext(file_name)[0].replace("event_tensor_", "")
    final_path = os.path.join(vis_save_dir, f"{name_stem}.png")
    cv2.imwrite(final_path, img_final)

    # -------------------------------
    # 5. 성능 지표 계산 (기존 로직 동일)
    # -------------------------------
    # ... (기존 지표 계산 코드 유지) ...
    num_gt = len(gt_lines) if gt_lines is not None else 0
    num_pred = len(pred_coords) if pred_coords is not None else 0

    img_f_scores = []
    for i in range(1, 100):
        tp, fp, fn = tp_fp_fn_func(pred_coords, gt_lines, thresh=i * 0.01)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        img_f_scores.append(f1)

    current_mf1 = np.mean(img_f_scores)
    current_hiou = hiou_func(pred_coords, gt_lines, (H, W))

    if folder_name not in stats_accumulator:
        stats_accumulator[folder_name] = {'hiou_sum': 0.0, 'mf1_sum': 0.0, 'gt_count_sum': 0, 'pred_count_sum': 0,
                                          'count': 0}

    stats_accumulator[folder_name]['hiou_sum'] += current_hiou
    stats_accumulator[folder_name]['mf1_sum'] += current_mf1
    stats_accumulator[folder_name]['gt_count_sum'] += num_gt
    stats_accumulator[folder_name]['pred_count_sum'] += num_pred
    stats_accumulator[folder_name]['count'] += 1

    return stats_accumulator

def process_eval_step(save_root, data_root, folder_name, file_name, model_alias,
                      img_tensor, pred_coords, gt_lines,
                      hiou_func, tp_fp_fn_func, H, W,
                      stats_accumulator):
    """
    [고화질 버전]
    1. vis_polarity 이미지를 원본 크기(346x260)로 로드
    2. 모델 예측 좌표(224x224)를 원본 비율에 맞춰 스케일링
    3. 원본 이미지 위에 직접 그리기 -> 선명도 저하 없음
    """
    # -------------------------------
    # 0. 해상도 설정
    # -------------------------------
    # 최종 저장할 목표 해상도 (원본 크기)
    TARGET_W = 346
    TARGET_H = 260

    # 모델이 예측한 기준 해상도 (224)
    MODEL_W = W
    MODEL_H = H

    # 확대 비율 계산 (예: 1.54배, 1.16배)
    scale_x = TARGET_W / MODEL_W
    scale_y = TARGET_H / MODEL_H

    # -------------------------------
    # 1. 이미지 로드 (vis_polarity)
    # -------------------------------
    vis_save_dir = os.path.join(save_root, str(folder_name), str(model_alias))
    os.makedirs(vis_save_dir, exist_ok=True)

    # 파일명 파싱
    name_stem = os.path.splitext(file_name)[0]
    name_stem = name_stem.replace("event_tensor_", "")

    # polarity_img_path = os.path.join(data_root, str(folder_name), "vis_polarity", f"vis_polarity_{name_stem}.png")
    polarity_img_path = os.path.join(data_root, str(folder_name), "rgb", f"dv_frame_{name_stem}.png")

    img_color = None

    # 1-1. vis_polarity 로드
    if os.path.exists(polarity_img_path):
        img_color = cv2.imread(polarity_img_path)

    # 1-2. 실패 시 Fallback (Tensor -> 흑백 -> 컬러)
    if img_color is None:
        if img_tensor.dim() == 3:
            img_np = img_tensor.detach().cpu().numpy().squeeze()
        else:
            img_np = img_tensor.detach().cpu().numpy()

        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
        img_np = (img_np * 255).astype(np.uint8)
        img_color = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    # -------------------------------
    # 2. 배경 이미지 크기 맞추기
    # -------------------------------
    # 로드한 이미지가 346x260이 아닐 수도 있으므로 강제로 맞춤
    # (이미 고화질이라면 화질 저하 없음)
    img_final = cv2.resize(img_color, (TARGET_W, TARGET_H))

    # -------------------------------
    # 3. 예측 라인 그리기 (좌표 스케일링 적용)
    # -------------------------------
    if pred_coords is not None:
        for line in pred_coords:
            try:
                # 224 좌표 꺼내기
                py1, px1, py2, px2 = line

                # ⭐️ 좌표를 원본 크기에 맞게 뻥튀기
                final_x1 = int(px1 * scale_x)
                final_y1 = int(py1 * scale_y)
                final_x2 = int(px2 * scale_x)
                final_y2 = int(py2 * scale_y)

                pt1 = (final_x1, final_y1)
                pt2 = (final_x2, final_y2)

                # 검정색 (0, 0, 0), 두께 2
                # 큰 이미지에 직접 그리므로 선이 아주 깔끔하게 나옵니다.
                cv2.line(img_final, pt1, pt2, (0, 128, 0), 4)

            except Exception:
                pass

                # -------------------------------
    # 4. 저장
    # -------------------------------
    final_path = os.path.join(vis_save_dir, f"{name_stem}.png")
    cv2.imwrite(final_path, img_final)

    # -------------------------------
    # 5. 성능 지표 계산 및 반환
    # -------------------------------
    num_gt = len(gt_lines) if gt_lines is not None else 0
    num_pred = len(pred_coords) if pred_coords is not None else 0

    img_f_scores = []
    for i in range(1, 100):
        tp, fp, fn = tp_fp_fn_func(pred_coords, gt_lines, thresh=i * 0.01)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        img_f_scores.append(f1)

    current_mf1 = np.mean(img_f_scores)
    current_hiou = hiou_func(pred_coords, gt_lines, (H, W))

    # 딕셔너리 누적
    if folder_name not in stats_accumulator:
        stats_accumulator[folder_name] = {
            'hiou_sum': 0.0, 'mf1_sum': 0.0,
            'gt_count_sum': 0, 'pred_count_sum': 0, 'count': 0
        }

    stats_accumulator[folder_name]['hiou_sum'] += current_hiou
    stats_accumulator[folder_name]['mf1_sum'] += current_mf1
    stats_accumulator[folder_name]['gt_count_sum'] += num_gt
    stats_accumulator[folder_name]['pred_count_sum'] += num_pred
    stats_accumulator[folder_name]['count'] += 1

    return stats_accumulator
def save_final_folder_stats(stats_accumulator, model_alias, save_root):  # 👈 save_root 인자 추가
    """
    폴더별 통계 저장 (사용자가 지정한 save_root 경로에 저장)
    """
    # 하드코딩된 "visualization" 대신 입력받은 save_root 사용
    csv_path = os.path.join(save_root, "All_Model_Folder_Stats.csv")

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        fieldnames = ['Model', 'Folder', 'Avg_HIoU', 'Avg_mF1', 'Avg_GT_Count', 'Avg_Pred_Count', 'Total_Samples']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for folder_name, data in stats_accumulator.items():
            cnt = data['count']
            # 0으로 나누기 방지
            avg_hiou = data['hiou_sum'] / cnt if cnt > 0 else 0
            avg_mf1 = data['mf1_sum'] / cnt if cnt > 0 else 0
            avg_gt = data['gt_count_sum'] / cnt if cnt > 0 else 0
            avg_pred = data['pred_count_sum'] / cnt if cnt > 0 else 0

            writer.writerow({
                'Model': model_alias,
                'Folder': folder_name,
                'Avg_HIoU': round(avg_hiou, 4),
                'Avg_mF1': round(avg_mf1, 4),
                'Avg_GT_Count': round(avg_gt, 2),
                'Avg_Pred_Count': round(avg_pred, 2),
                'Total_Samples': cnt
            })

    print(f"📊 폴더별 통계 저장 완료: {csv_path}")


# ▍정렬 기반 patch vector 시각화
# sorted_pid, sort_idx = pid.sort()
# X_flat = patch_id_map.view(-1)
# sorted_vals = X_flat[sort_idx]
# patch_vectors = sorted_vals.view(num_patches, patch * patch)  # (70, 1024)
# print("패치 임베딩 shape:", patch_vectors.shape)

