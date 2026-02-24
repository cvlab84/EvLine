import torch
import numpy as np
import math


# =============================================================================
# 1. Hough Transform & Patch Mapping Utilities
# =============================================================================

def build_hough_bank(image_height, image_width, theta_res=1.8, rho_res=math.sqrt(2) * 2):
    """
    Generate a Hough bank tensor for all pixels in an image.
    This serves as the geometric prior for mapping event spaces to Hough spaces.
    """
    # 1. Define theta values
    theta = np.arange(0, 180, theta_res)  # [0, 180)
    T = len(theta)
    cos_t = np.cos(np.deg2rad(theta))
    sin_t = np.sin(np.deg2rad(theta))

    # 2. Calculate HALF maximum rho (논문 스타일: [-L/2, +L/2])
    diag_len = np.hypot(image_height, image_width)
    rho_max = diag_len / 2
    R = int(np.ceil(diag_len / rho_res))
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


def make_patch_id_map(H, W, patch, device='cpu'):
    """
    Generate a Patch ID map for spatial to patch-wise aggregation.
    """
    n_h, n_w = H // patch, W // patch
    y = torch.arange(H, device=device) // patch
    x = torch.arange(W, device=device) // patch
    return (y.view(H, 1) * n_w + x.view(1, W)).long()


def hough_embedd_chunk(X, patch_id_map, hbank, num_patches, chunk=1000):
    """
    Memory-efficient version of Hough embedding using chunk-wise multiplication.
    """
    B, _, H, W = X.shape
    R, T = hbank.shape[-2:]
    out = torch.zeros(B, num_patches, R, T, device=X.device)
    patch_ids = patch_id_map.view(-1)
    hbank_flat = hbank.view(H * W, R * T)

    for b in range(B):
        X_flat = X[b].view(-1)  # (H*W,)
        out_b = torch.zeros(num_patches, R, T, device=X.device)
        for start in range(0, H * W, chunk):
            end = min(start + chunk, H * W)
            contrib = X_flat[start:end].unsqueeze(1) * hbank_flat[start:end]  # (chunk, R*T)
            contrib = contrib.view(-1, R, T)
            out_b.index_add_(0, patch_ids[start:end], contrib)
        out[b] = out_b

    # Normalization & Flatten
    amax = out.amax(dim=(1, 2, 3), keepdim=True)
    out = out / (amax + 1e-6)
    out = out.view(B, num_patches, -1)  # (B, N, R*T)

    return out


# =============================================================================
# 2. Decoding & Evaluation Utilities
# =============================================================================

def get_boundary_point(y, x, angle, H, W):
    """
    Given point y,x with angle, return two points in image boundary.
    """
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
        if 0 <= y - k * x < H:  # left
            if point1 is None:
                point1 = (0, int(y - k * x))
            elif point2 is None:
                point2 = (0, int(y - k * x))
                if point2 == point1: point2 = None

        if 0 <= k * (W - 1) + y - k * x < H:  # right
            if point1 is None:
                point1 = (W - 1, int(k * (W - 1) + y - k * x))
            elif point2 is None:
                point2 = (W - 1, int(k * (W - 1) + y - k * x))
                if point2 == point1: point2 = None

        if 0 <= x - y / k < W:  # top
            if point1 is None:
                point1 = (int(x - y / k), 0)
            elif point2 is None:
                point2 = (int(x - y / k), 0)
                if point2 == point1: point2 = None

        if 0 <= x - y / k + (H - 1) / k < W:  # bottom
            if point1 is None:
                point1 = (int(x - y / k + (H - 1) / k), H - 1)
            elif point2 is None:
                point2 = (int(x - y / k + (H - 1) / k), H - 1)
                if point2 == point1: point2 = None

        if point2 is None:
            point2 = point1

    return point1, point2


def reverse_mapping(point_list, numRho=112, theta_res=1.8, rho_res=math.sqrt(2) * 2, size=(224, 224)):
    """
    Map coordinates from Hough space (theta index, rho index) back to
    image space (H, W) line endpoints.
    """
    H, W = size
    itheta = np.deg2rad(theta_res)
    b_points = []

    for (ri, thetai) in point_list:
        theta = thetai * itheta
        r = (ri - numRho // 2) * rho_res

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        if abs(sin_t) < 1e-6:
            # Vertical line
            x = int(np.round(r / cos_t + W / 2))
            p1, p2 = (0, x), (H - 1, x)
        else:
            # Slanted line
            angle = np.arctan(-cos_t / sin_t)
            y0 = np.round(r / sin_t + (W * cos_t) / (2 * sin_t) + H / 2)
            p1, p2 = get_boundary_point(int(y0), 0, angle, H, W)

        if p1 is not None and p2 is not None:
            b_points.append((p1[1], p1[0], p2[1], p2[0]))

    return b_points