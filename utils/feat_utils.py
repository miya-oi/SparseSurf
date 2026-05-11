import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union
from collections import OrderedDict


def bin_op_reduce(lst, func):
    result = lst[0]
    for i in range(1, len(lst)):
        result = func(result, lst[i])
    return result


def idx_world2cam(idx_world_homo, cam):
    """nhw41 -> nhw41"""
    idx_cam_homo = cam[:,0:1,...].unsqueeze(1) @ idx_world_homo  # nhw41
    idx_cam_homo = idx_cam_homo / (idx_cam_homo[...,-1:,:]+1e-9)   # nhw41
    return idx_cam_homo


def idx_cam2img(idx_cam_homo, cam):
    """nhw41 -> nhw31"""
    idx_cam = idx_cam_homo[...,:3,:] / (idx_cam_homo[...,3:4,:]+1e-9)  # nhw31
    idx_img_homo = cam[:,1:2,:3,:3].unsqueeze(1) @ idx_cam  # nhw31
    idx_img_homo = idx_img_homo / (idx_img_homo[...,-1:,:]+1e-9)
    return idx_img_homo


def normalize_for_grid_sample(input_, grid):
    size = torch.tensor(input_.size())[2:].flip(0).to(grid.dtype).to(grid.device).view(1,1,1,-1)  # 111N
    grid_n = grid / size
    grid_n = (grid_n * 2 - 1).clamp(-1.1, 1.1)
    return grid_n


def get_in_range(grid):
    """after normalization, keepdim=False"""
    masks = []
    for dim in range(grid.size()[-1]):
        masks += [grid[..., dim]<=1, grid[..., dim]>=-1]
    in_range = bin_op_reduce(masks, torch.min).to(grid.dtype)
    return in_range


def load_pair(file: str, min_views: int=None):
    with open(file) as f:
        lines = f.readlines()
    n_cam = int(lines[0])
    pairs = {}
    img_ids = []
    for i in range(1, 1+2*n_cam, 2):
        pair = []
        score = []
        img_id = lines[i].strip()
        pair_str = lines[i+1].strip().split(' ')
        n_pair = int(pair_str[0])
        if min_views is not None and n_pair < min_views: continue
        for j in range(1, 1+2*n_pair, 2):
            pair.append(pair_str[j])
            score.append(float(pair_str[j+1]))
        img_ids.append(img_id)
        pairs[img_id] = {'id': img_id, 'index': i//2, 'pair': pair, 'score': score}
    pairs['id_list'] = img_ids
    return pairs


class ListModule(nn.Module):
    def __init__(self, modules: Union[List, OrderedDict]):
        super(ListModule, self).__init__()
        if isinstance(modules, OrderedDict):
            iterable = modules.items()
        elif isinstance(modules, list):
            iterable = enumerate(modules)
        else:
            raise TypeError('modules should be OrderedDict or List.')
        for name, module in iterable:
            if not isinstance(module, nn.Module):
                module = ListModule(module)
            if not isinstance(name, str):
                name = str(name)
            self.add_module(name, module)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dim=2):
        super(BasicBlock, self).__init__()

        self.conv_fn = nn.Conv2d if dim == 2 else nn.Conv3d
        self.bn_fn = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
        # self.bn_fn = nn.GroupNorm

        self.conv1 = self.conv3x3(inplanes, planes, stride)
        # nn.init.xavier_uniform_(self.conv1.weight)
        self.bn1 = self.bn_fn(planes)
        # nn.init.constant_(self.bn1.weight, 1)
        # nn.init.constant_(self.bn1.bias, 0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.conv3x3(planes, planes)
        # nn.init.xavier_uniform_(self.conv2.weight)
        self.bn2 = self.bn_fn(planes)
        # nn.init.constant_(self.bn2.weight, 0)
        # nn.init.constant_(self.bn2.bias, 0)
        self.downsample = downsample
        self.stride = stride

    def conv1x1(self, in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return self.conv_fn(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def conv3x3(self, in_planes, out_planes, stride=1):
        """3x3 convolution with padding"""
        return self.conv_fn(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def _make_layer(inplanes, block, planes, blocks, stride=1, dim=2):
    downsample = None
    conv_fn = nn.Conv2d if dim==2 else nn.Conv3d
    bn_fn = nn.BatchNorm2d if dim==2 else nn.BatchNorm3d
    # bn_fn = nn.GroupNorm
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv_fn(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
            bn_fn(planes * block.expansion)
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample, dim=dim))
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(inplanes, planes, dim=dim))

    return nn.Sequential(*layers)


class UNet(nn.Module):

    def __init__(self, inplanes: int, enc: int, dec: int, initial_scale: int,
                 bottom_filters: List[int], filters: List[int], head_filters: List[int],
                 prefix: str, dim: int=2):
        super(UNet, self).__init__()

        conv_fn = nn.Conv2d if dim==2 else nn.Conv3d
        bn_fn = nn.BatchNorm2d if dim==2 else nn.BatchNorm3d
        # bn_fn = nn.GroupNorm
        deconv_fn = nn.ConvTranspose2d if dim==2 else nn.ConvTranspose3d
        current_scale = initial_scale
        idx = 0
        prev_f = inplanes

        self.bottom_blocks = OrderedDict()
        for f in bottom_filters:
            block = _make_layer(prev_f, BasicBlock, f, enc, 1 if idx==0 else 2, dim=dim)
            self.bottom_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale *= 2
            prev_f = f
        self.bottom_blocks = ListModule(self.bottom_blocks)

        self.enc_blocks = OrderedDict()
        for f in filters:
            block = _make_layer(prev_f, BasicBlock, f, enc, 1 if idx == 0 else 2, dim=dim)
            self.enc_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale *= 2
            prev_f = f
        self.enc_blocks = ListModule(self.enc_blocks)

        self.dec_blocks = OrderedDict()
        for f in filters[-2::-1]:
            block = [
                deconv_fn(prev_f, f, 3, 2, 1, 1, bias=False),
                conv_fn(2*f, f, 3, 1, 1, bias=False),
            ]
            if dec > 0:
                block.append(_make_layer(f, BasicBlock, f, dec, 1, dim=dim))
            # nn.init.xavier_uniform_(block[0].weight)
            # nn.init.xavier_uniform_(block[1].weight)
            self.dec_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale //= 2
            prev_f = f
        self.dec_blocks = ListModule(self.dec_blocks)

        self.head_blocks = OrderedDict()
        for f in head_filters:
            block = [
                deconv_fn(prev_f, f, 3, 2, 1, 1, bias=False)
            ]
            if dec > 0:
                block.append(_make_layer(f, BasicBlock, f, dec, 1, dim=dim))
            block = nn.Sequential(*block)
            # nn.init.xavier_uniform_(block[0])
            self.head_blocks[f'{prefix}{current_scale}_{idx}'] = block
            idx += 1
            current_scale //= 2
            prev_f = f
        self.head_blocks = ListModule(self.head_blocks)

    def forward(self, x, multi_scale=1):
        for b in self.bottom_blocks:
            x = b(x)
        enc_out = []
        for b in self.enc_blocks:
            x = b(x)
            enc_out.append(x)
        dec_out = [x]
        for i, b in enumerate(self.dec_blocks):
            if len(b) == 3: deconv, post_concat, res = b
            elif len(b) == 2: deconv, post_concat = b
            x = deconv(x)
            x = torch.cat([x, enc_out[-2-i]], 1)
            x = post_concat(x)
            if len(b) == 3: x = res(x)
            dec_out.append(x)
        for b in self.head_blocks:
            x = b(x)
            dec_out.append(x)
        if multi_scale == 1: return x
        else: return dec_out[-multi_scale:]


class FeatExt(nn.Module):

    def __init__(self):
        super(FeatExt, self).__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.unet = UNet(16, 2, 1, 2, [], [32, 64, 128], [], '2d', 2)
        self.final_conv_1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.final_conv_2 = nn.Conv2d(64, 32, 3, 1, 1, bias=False)
        self.final_conv_3 = nn.Conv2d(32, 32, 3, 1, 1, bias=False)

        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vismvsnet.pt")
        feat_ext_dict = {
            k[16:]: v
            for k, v in torch.load(weights_path, map_location="cpu")["state_dict"].items()
            if k.startswith("module.feat_ext")
        }
        self.load_state_dict(feat_ext_dict)

    def forward(self, x):
        out = self.init_conv(x)
        out1, out2, out3 = self.unet(out, multi_scale=3)
        return self.final_conv_1(out1), self.final_conv_2(out2), self.final_conv_3(out3)


def _pack_feature_camera(viewpoint_cam):
    packed = viewpoint_cam.intrinsic.new_zeros((2, 4, 4))
    packed[0] = viewpoint_cam.world_view_transform.T
    packed[1] = viewpoint_cam.intrinsic
    return packed


def _stack_feature_cameras(src_viewpoint_stack):
    camera_blocks = [_pack_feature_camera(src_viewpoint_cam).unsqueeze(0) for src_viewpoint_cam in src_viewpoint_stack]
    return torch.cat(camera_blocks, dim=0).unsqueeze(0)


def _split_masked_tracks(mask, batch_size):
    flat_mask = mask.view(batch_size, -1)
    counts = flat_mask.sum(-1)
    offsets = [0] + counts.cumsum(0).tolist()
    spans = [slice(offsets[idx], offsets[idx + 1]) for idx in range(len(offsets) - 1)]
    return spans


def _project_world_points(world_points, camera_pack):
    points_h = world_points.view(1, -1, 1, 3, 1)
    points_h = torch.cat([points_h, torch.ones_like(points_h[..., -1:, :])], dim=-2)
    projected = idx_cam2img(idx_world2cam(points_h, camera_pack), camera_pack)
    return projected[..., :2, 0]


def _build_frontmost_visibility(pixel_tracks, camera_pack, world_points):
    source_centers = torch.inverse(camera_pack[1:, 0, ...])[:, :3, 3]
    source_distances = torch.norm(world_points[None] - source_centers[:, None], dim=-1)
    rounded_pixels = pixel_tracks[1:].squeeze(2).round().long()

    visible_tracks = []
    for src_idx in range(rounded_pixels.shape[0]):
        _, near_to_far = torch.sort(source_distances[src_idx])
        ordered_pixels = rounded_pixels[src_idx, near_to_far]
        _, duplicate_counts = torch.unique(ordered_pixels, sorted=False, return_counts=True, dim=0)
        first_hit = torch.cumsum(torch.cat([duplicate_counts.new_zeros(1), duplicate_counts]), dim=0)[:-1]
        keep_mask = torch.zeros_like(near_to_far)
        keep_mask[first_hit] = 1
        _, restore_order = torch.sort(near_to_far)
        visible_tracks.append(keep_mask[restore_order])

    return torch.stack(visible_tracks, dim=0) > 0.5


def _sample_cross_view_features(target_feat, source_feat, pixel_tracks, scale):
    sample_device = pixel_tracks.device
    feature_bank = torch.cat([target_feat, source_feat], dim=0).to(sample_device)
    normalized_tracks = normalize_for_grid_sample(feature_bank, pixel_tracks.to(sample_device) / scale)
    in_frame = get_in_range(normalized_tracks)
    paired_support = (in_frame[:1] * in_frame[1:]).unsqueeze(1) > 0.5
    sampled_features = F.grid_sample(
        feature_bank,
        normalized_tracks,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False,
    )
    return sampled_features, paired_support


def _reduce_sampled_feature_gap(sampled_features, paired_support, visible_tracks):
    feature_norm = sampled_features.norm(dim=1, keepdim=True)
    cosine_tracks = (sampled_features[:1] * sampled_features[1:]).sum(dim=1, keepdim=True)
    cosine_tracks = cosine_tracks / feature_norm[:1].clamp(min=1e-9) / feature_norm[1:].clamp(min=1e-9)
    feature_gap = (1 - cosine_tracks).abs()
    stable_tracks = feature_gap < 0.5
    visible_tracks = visible_tracks.reshape_as(paired_support)
    return (feature_gap * paired_support * stable_tracks * visible_tracks).mean()


def _measure_projected_feature_gap(active_points, target_feat, target_cam, source_feat, source_cams, mask, scale):
    if mask.sum() == 0:
        return active_points.new_tensor(0.0)

    active_spans = _split_masked_tracks(mask, target_feat.size(0))
    per_view_losses = []

    for view_idx, active_span in enumerate(active_spans):
        if active_span.start >= active_span.stop:
            per_view_losses.append(active_points.new_tensor(0.0))
            continue

        visible_points = active_points[active_span]
        camera_pack = torch.cat([target_cam[view_idx:view_idx + 1], source_cams[view_idx]], dim=0)
        pixel_tracks = _project_world_points(visible_points, camera_pack)
        visible_tracks = _build_frontmost_visibility(pixel_tracks, camera_pack, visible_points)
        sampled_features, paired_support = _sample_cross_view_features(
            target_feat[view_idx:view_idx + 1],
            source_feat[view_idx],
            pixel_tracks,
            scale,
        )
        per_view_losses.append(_reduce_sampled_feature_gap(sampled_features, paired_support, visible_tracks))

    return sum(per_view_losses) / len(per_view_losses)


def _feature_alignment_schedule(resolution):
    if resolution == 2:
        return ((0, 1.0, 1), (1, 0.5, 2))
    return ((0, 1.0, 2), (1, 0.5, 4))


def compute_reference_view_feature_penalty(pts_world, viewpoint_cam, src_viewpoint_stack, mask, resolution=2, use_mask=True):
    target_cam = _pack_feature_camera(viewpoint_cam).unsqueeze(0)
    source_cams = _stack_feature_cameras(src_viewpoint_stack)

    active_mask = mask if use_mask else torch.ones_like(pts_world[:, 0])
    selected_points = pts_world[active_mask.bool()]

    total_penalty = 0.0
    total_weight = 0.0

    for feat_level, level_weight, sampling_scale in _feature_alignment_schedule(resolution):
        source_feats = [src_viewpoint_cam.feat[feat_level].unsqueeze(0) for src_viewpoint_cam in src_viewpoint_stack]
        source_feat_pack = torch.cat(source_feats, dim=0).unsqueeze(0)
        target_feat_pack = viewpoint_cam.feat[feat_level].unsqueeze(0)
        total_penalty += level_weight * _measure_projected_feature_gap(
            selected_points,
            target_feat_pack,
            target_cam,
            source_feat_pack,
            source_cams,
            active_mask.long(),
            scale=sampling_scale,
        )
        total_weight += level_weight

    return total_penalty / total_weight


def get_feat_loss(pts_world, viewpoint_cam, src_viewpoint_stack, mask, resolution=2, use_mask=True):
    return compute_reference_view_feature_penalty(
        pts_world,
        viewpoint_cam,
        src_viewpoint_stack,
        mask,
        resolution=resolution,
        use_mask=use_mask,
    )

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
