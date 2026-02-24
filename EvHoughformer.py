import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


# =============================================================================
# 1. Backbones: Adapted for 1-channel event count maps & feature resolution
# =============================================================================

class MobileNetV2Backbone(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=None)

        # 1-channel input adaptation for event maps
        old_conv = mobilenet.features[0][0]
        mobilenet.features[0][0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        self.features = mobilenet.features

        # Apply dilation to maintain spatial resolution for dense prediction
        for i in range(14, 19):
            for m in self.features[i].modules():
                if isinstance(m, nn.Conv2d):
                    if m.stride == (2, 2):
                        m.stride = (1, 1)
                    if m.kernel_size == (3, 3):
                        m.dilation = (2, 2)
                        m.padding = (2, 2)

    def forward(self, x):
        return self.features(x)  # Output channels: 1280


class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        resnet = models.resnet50(weights=None)

        # 1-channel input adaptation for event maps
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Apply dilation to layer4 to maintain spatial resolution
        self.layer4[0].conv2.stride = (1, 1)
        self.layer4[0].downsample[0].stride = (1, 1)

        for m in self.layer4.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.dilation = (2, 2)
                    m.padding = (2, 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x  # Output channels: 2048


# =============================================================================
# 2. Transformer Components (Event Stream)
# =============================================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, tr_dim, tr_num_heads):
        super().__init__()
        assert tr_dim % tr_num_heads == 0, "tr_dim must be divisible by tr_num_heads"
        self.num_heads = tr_num_heads
        self.head_dim = tr_dim // tr_num_heads

        self.q_proj = nn.Linear(tr_dim, tr_dim)
        self.k_proj = nn.Linear(tr_dim, tr_dim)
        self.v_proj = nn.Linear(tr_dim, tr_dim)
        self.o_proj = nn.Linear(tr_dim, tr_dim)

    def forward(self, x):
        B, N, D = x.shape
        H, Hd = self.num_heads, self.head_dim

        Q = self.q_proj(x).view(B, N, H, Hd).transpose(1, 2)
        K = self.k_proj(x).view(B, N, H, Hd).transpose(1, 2)
        V = self.v_proj(x).view(B, N, H, Hd).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Hd ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.o_proj(out)

        return out, attn_weights


class Transformer(nn.Module):
    def __init__(self, tr_dim, tr_num_heads):
        super().__init__()
        self.attn = MultiHeadSelfAttention(tr_dim, tr_num_heads)
        self.norm1 = nn.LayerNorm(tr_dim)
        self.ff = nn.Sequential(
            nn.Linear(tr_dim, tr_dim * 4),
            nn.ReLU(),
            nn.Linear(tr_dim * 4, tr_dim)
        )
        self.norm2 = nn.LayerNorm(tr_dim)

    def forward(self, x):
        attn_out, attn_weights = self.attn(x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x, attn_weights


# =============================================================================
# 3. Hough Stream Components
# =============================================================================

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_dilations=4):
        super().__init__()
        self.num_dilations = num_dilations
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d)
            for d in range(1, num_dilations + 1)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm2d(out_channels, affine=True)
            for _ in range(num_dilations)
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        branch_outs = []
        for conv, norm in zip(self.branches, self.norms):
            out = self.relu(norm(conv(x)))
            branch_outs.append(out)
        return torch.cat(branch_outs, dim=1)


class HoughStream(nn.Module):
    def __init__(self, R, T, tr_num_heads, N, hough_stack_depth=3, hough_num_dilations=4):
        super().__init__()
        self.R, self.T = R, T
        self.num_heads = tr_num_heads
        self.N = N
        self.stack_depth = hough_stack_depth
        self.num_dilations = hough_num_dilations

        # Compress channels derived from multi-head attention
        self.input_compress = nn.Conv2d(tr_num_heads * N, N, kernel_size=1)

        self.stacks = nn.ModuleList([
            DilatedConvBlock(
                in_channels=N if i == 0 else self.num_dilations * N,
                out_channels=N,
                num_dilations=self.num_dilations
            )
            for i in range(self.stack_depth)
        ])

    def forward(self, x_hough, attn_weights):
        B, N, RT = x_hough.shape
        R, T = self.R, self.T
        H = attn_weights.shape[1]

        # Inject event-driven attention into Hough space
        x_weighted = torch.einsum('bhnm,bmr->bhnr', attn_weights, x_hough)
        x_weighted = x_weighted.view(B, H * N, R, T)

        x_weighted = self.input_compress(x_weighted)

        for block in self.stacks:
            x_weighted = block(x_weighted)

        x_weighted = x_weighted.view(B, self.num_dilations, N, R, T)
        x = x_weighted.mean(dim=1)
        x = x.view(B, N, R * T)
        return x


# =============================================================================
# 4. Cross Attention Regression Head
# =============================================================================

class Crossatt(nn.Module):
    def __init__(self, R, T, N, dim):
        super().__init__()
        self.R, self.T, self.N = R, T, N
        self.dim = dim

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(R * T, dim)
        self.proj = nn.Conv2d(N, 1, kernel_size=1)

    def forward(self, x_hough, x_seq):
        B, N, RT = x_hough.shape
        R, T = self.R, self.T

        K = self.k_proj(x_hough)
        Q = self.q_proj(x_seq)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        x_weighted = torch.matmul(attn, x_hough)
        x_weighted = x_weighted.view(B, N, R, T)
        heatmap = self.proj(x_weighted)

        return heatmap, attn


# =============================================================================
# 5. Main Architecture: EvHoughformer
# =============================================================================

class EvHoughformer(nn.Module):
    def __init__(self, backbone_type='mobilenetv2', tr_dim=128, tr_num_heads=4, R=112, T=100,
                 dual_num_stacks=5, hough_stack_depth=1, hough_num_dilations=3):
        super().__init__()

        # Select Backbone
        if backbone_type == 'mobilenetv2':
            self.backbone = MobileNetV2Backbone()
            in_channels = 1280
        elif backbone_type == 'resnet50':
            self.backbone = ResNet50Backbone()
            in_channels = 2048
        else:
            raise ValueError("backbone_type must be either 'mobilenetv2' or 'resnet50'")

        self.input_proj = nn.Conv2d(in_channels, tr_dim, kernel_size=1)

        self.N = 196  # 14x14 patches based on 224x224 input
        self.pos_embed = nn.Parameter(torch.zeros(1, self.N, tr_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.upper_stacks = nn.ModuleList([
            Transformer(tr_dim, tr_num_heads) for _ in range(dual_num_stacks)
        ])
        self.lower_stacks = nn.ModuleList([
            HoughStream(R, T, tr_num_heads, self.N, hough_stack_depth, hough_num_dilations)
            for _ in range(dual_num_stacks)
        ])

        self.pred_head = Crossatt(R, T, self.N, dim=tr_dim)

    def forward(self, x_raw, x_hough):
        feat = self.backbone(x_raw)

        # Interpolate if feature map size does not match expected patch size (14x14)
        if feat.shape[-1] != 14:
            feat = F.interpolate(feat, size=(14, 14), mode='bilinear', align_corners=False)

        feat = self.input_proj(feat)
        x_seq = feat.flatten(2).transpose(1, 2)
        x_seq = x_seq + self.pos_embed

        for upper, lower in zip(self.upper_stacks, self.lower_stacks):
            x_seq, attn = upper(x_seq)
            x_hough = lower(x_hough, attn)

        heatmap, acc = self.pred_head(x_hough, x_seq)

        # We return only heatmap for standard inference.
        # (Internal sequences can be returned if needed for loss constraints)
        return heatmap


# =============================================================================
# Quick Start / Preliminary Validation
# =============================================================================
if __name__ == "__main__":
    # Ensure these functions are imported correctly from your hough_utils.py during actual usage
    # from hough_utils import build_hough_bank, make_patch_id_map, hough_embedd_chunk

    print("=== EvHoughformer Preliminary Test ===")

    # Environment Setup
    H, W = 224, 224
    patch = 16
    theta_res = 1.8
    rho_res = math.sqrt(2) * 2
    num_patches = (H // patch) * (W // patch)
    B = 2  # Batch size

    print("\n1. Initializing Models (CPU mode)...")
    # Initialize both variants described in the paper
    model_mbv2 = EvHoughformer(backbone_type='mobilenetv2', tr_dim=128, tr_num_heads=4, R=112, T=100, dual_num_stacks=5)
    model_rn50 = EvHoughformer(backbone_type='resnet50', tr_dim=128, tr_num_heads=4, R=112, T=100, dual_num_stacks=5)

    # Verify parameter counts
    params_mbv2 = sum(p.numel() for p in model_mbv2.parameters() if p.requires_grad)
    params_rn50 = sum(p.numel() for p in model_rn50.parameters() if p.requires_grad)
    print(f" - MobileNetV2 Params : {params_mbv2 / 1e6:.1f} M")
    print(f" - ResNet50 Params    : {params_rn50 / 1e6:.1f} M")

    print("\n2. Building Synthetic Data for Testing...")
    # NOTE: Uncomment below when hough_utils is available
    # hbank, rho_list, theta_list, coords = build_hough_bank(H, W, theta_res, rho_res)
    # hbank = torch.from_numpy(hbank)
    # patch_id_map = make_patch_id_map(H, W, patch)

    sample_event_map = torch.rand(B, 1, H, W) / 3.0
    # dummy_hough_embed = hough_embedd_chunk(sample_event_map, patch_id_map, hbank, num_patches, chunk=1000)

    # Standalone testing without hough_utils
    sample_hough_embed = torch.rand(B, num_patches, 112 * 100)

    print("\n3. Running Forward Pass...")
    out_heatmap_mbv2 = model_mbv2(sample_event_map, sample_hough_embed)
    out_heatmap_rn50 = model_rn50(sample_event_map, sample_hough_embed)

    print(f" - MobileNetV2 Output Shape : {out_heatmap_mbv2.shape} (Expected: {B}, 1, 112, 100)")
    print(f" - ResNet50 Output Shape    : {out_heatmap_rn50.shape} (Expected: {B}, 1, 112, 100)")
    print("\n✅ Core architecture executed successfully!")