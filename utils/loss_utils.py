#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torchvision.transforms as transforms
from utils.graphics_utils import patch_offsets, patch_warp, batch_patch_warp
import numpy as np

transform1 = transforms.CenterCrop((576, 768))
transform2 = transforms.CenterCrop((544, 736))

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def loss_depth_smoothness(depth, img):
    depth = depth.unsqueeze(0)
    img = img.unsqueeze(0)
    img_grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    img_grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]

    weight_x = torch.exp(-torch.abs(img_grad_x).mean(1, keepdim=True))
    weight_y = torch.exp(-torch.abs(img_grad_y).mean(1, keepdim=True))

    loss = (((depth[:, :, :, :-1] - depth[:, :, :, 1:]).abs() * weight_x).sum() +
            ((depth[:, :, :-1, :] - depth[:, :, 1:, :]).abs() * weight_y).sum()) / \
           (weight_x.sum() + weight_y.sum())
    return loss

def train_view_fa_loss(
    pts_world,                 # [M,3]  target帧的3D点（世界坐标），例如 surf_points[mask]
    feat_t,                    # [1,C,H,W] target feature（与 get_feat_loss_corr 一致）
    cam_t,                     # [1,2,4,4] target cam pack（同你现有cam）
    feat_s,                    # [1,C,H,W] source feature（单个src）
    cam_s,                     # [1,2,4,4] source cam pack（单个src）
    scale=2,                   # 同你工程：grid / scale
    valid_mask=None,           # [M] or None，用于额外筛选（可选）
    eps=1e-9
):

    # 1) 把 pts_world 投影到 target + source（与 get_feat_loss_corr 同链路）
    pts = pts_world.view(1, -1, 1, 3, 1)  # [1,M,1,3,1]
    pts = torch.cat([pts, torch.ones_like(pts[..., -1:, :])], dim=-2)  # [1,M,1,4,1]

    cam_pack = torch.cat([cam_t, cam_s], dim=0)  # [2,2,4,4]  (v=2)
    pts_img = idx_cam2img(idx_world2cam(pts, cam_pack), cam_pack)      # [2,M,1,3,1]
    grid = pts_img[..., :2, 0]  # [2,M,1,2]
    grid_t = grid[:1]           # [1,M,1,2]
    grid_s = grid[1:]           # [1,M,1,2]

    # 2) grid normalize（沿用你的 normalize_for_grid_sample）
    grid_tn = normalize_for_grid_sample(feat_t, grid_t / scale)  # [1,M,1,2]
    grid_sn = normalize_for_grid_sample(feat_s, grid_s / scale)  # [1,M,1,2]

    in_t = get_in_range(grid_tn)  # [1,M,1]
    in_s = get_in_range(grid_sn)  # [1,M,1]
    valid = (in_t * in_s) > 0.5   # [1,M,1]

    # 可选：叠加你自己的 mask（比如高置信点）
    if valid_mask is not None:
        # valid_mask 期望是 [M] 或 [1,M]，这里统一成 [1,M,1]
        vm = valid_mask.view(1, -1, 1) > 0.5
        valid = valid & vm

    # 3) 从两张特征图采样：得到 F_t(u_t), F_s(u_s)
    ft = F.grid_sample(feat_t, grid_tn, mode="bilinear", padding_mode="zeros", align_corners=False)  # [1,C,M,1]
    fs = F.grid_sample(feat_s, grid_sn, mode="bilinear", padding_mode="zeros", align_corners=False)  # [1,C,M,1]

    # 4) pixel-wise cosine
    dot = (ft * fs).sum(dim=1, keepdim=True)                    # [1,1,M,1]
    nt = ft.norm(dim=1, keepdim=True).clamp(min=eps)
    ns = fs.norm(dim=1, keepdim=True).clamp(min=eps)
    cos = dot / (nt * ns)                                       # [1,1,M,1]

    loss_map = (1.0 - cos).abs()                                # [1,1,M,1]

    valid_f = valid.float().unsqueeze(1)                        # [1,1,M,1]
    denom = valid_f.sum().clamp(min=1.0)
    loss = (loss_map * valid_f).sum() / denom
    return loss