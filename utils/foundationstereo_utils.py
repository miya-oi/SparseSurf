import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from utils.graphics_utils import fov2focal
import torchvision

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
FOUNDATION_STEREO_DIR = os.path.join(MODULE_DIR, "FoundationStereo")
FOUNDATION_STEREO_CORE_DIR = os.path.join(FOUNDATION_STEREO_DIR, "core")
DEFAULT_FOUNDATION_STEREO_CKPT = os.path.join(
    FOUNDATION_STEREO_DIR, "pretrained_models", "23-51-11", "model_best_bp2-001.pth"
)

for path in (FOUNDATION_STEREO_DIR, FOUNDATION_STEREO_CORE_DIR):
    if path not in sys.path:
        sys.path.append(path)

from foundation_stereo import FoundationStereo

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_foundation_stereo_ckpt(ckpt_path=None):
    candidates = []
    if ckpt_path:
        candidates.append(ckpt_path)

    primary_env_ckpt = os.environ.get("FOUNDATION_STEREO_CKPT")
    if primary_env_ckpt:
        candidates.append(primary_env_ckpt)

    candidates.append(DEFAULT_FOUNDATION_STEREO_CKPT)

    checked_paths = []
    for candidate in candidates:
        resolved = os.path.abspath(os.path.expanduser(candidate))
        checked_paths.append(resolved)
        if os.path.isfile(resolved):
            return resolved

    checked = "\n".join(checked_paths)
    raise FileNotFoundError(
        "FoundationStereo checkpoint not found.\n"
        "Set --foundation_stereo_ckpt or FOUNDATION_STEREO_CKPT.\n"
        f"Tried:\n{checked}"
    )


def load_foundation_stereo_args(ckpt_path):
    cfg_path = os.path.join(os.path.dirname(ckpt_path), "cfg.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing FoundationStereo cfg.yaml next to checkpoint: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"

    cfg.update(
        {
            "ckpt_dir": ckpt_path,
            "scale": 1,
            "hiera": 0,
            "z_far": 10,
            "valid_iters": 32,
            "get_pc": 1,
            "remove_invisible": 1,
            "denoise_cloud": 1,
            "denoise_nb_points": 30,
            "denoise_radius": 0.03,
        }
    )
    return OmegaConf.create(cfg)


class InputPadder:
    """Pads images such that dimensions are divisible by 8."""

    def __init__(self, dims, mode="sintel", divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == "sintel":
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        crop = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., crop[0]:crop[1], crop[2]:crop[3]]


def normalize_image(img):
    """Normalize RGB images in range [0, 255]."""

    transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        inplace=False,
    )
    return transform(img / 255.0).contiguous()


class Foundation:
    def __init__(self, ckpt_path=None):
        resolved_ckpt = resolve_foundation_stereo_ckpt(ckpt_path)
        args = load_foundation_stereo_args(resolved_ckpt)
        if "iraftstereo_rvc" in resolved_ckpt:
            args.context_norm = "instance"
        if DEVICE == "cuda":
            self.model = torch.nn.DataParallel(FoundationStereo(args), device_ids=[0])
            self.model = self.model.module
        else:
            self.model = FoundationStereo(args)
        self.model.load_state_dict(torch.load(resolved_ckpt, map_location="cpu")["model"])
        self.model.to(DEVICE)
        self.model.eval()

    def get_feature(self, x):
        x = x.clone()
        x *= 255
        padder = InputPadder(x.shape, divis_by=32)
        x = padder.pad(x)[0]
        ori_resolution = (x.shape[2], x.shape[3])

        x = normalize_image(x)
        x = self.model.feature.stem(x)
        out = self.model.feature.stages[0](x)
        out = F.interpolate(out, size=ori_resolution, mode="bilinear", align_corners=True)
        out = padder.unpad(out)
        return out

    def predict_disparity(self, left_image, right_image, flip=False):
        with torch.no_grad():
            left_image = left_image.clone()
            right_image = right_image.clone()
            left_image *= 255
            right_image *= 255
            padder = InputPadder(left_image.shape, divis_by=32)
            left_image, right_image = padder.pad(left_image, right_image)
            if flip:
                image1_flip = torch.flip(left_image, dims=[3])
                image2_flip = torch.flip(right_image, dims=[3])
                with torch.cuda.amp.autocast(True):
                    flow_up_flip = self.model.forward(image2_flip, image1_flip, iters=32, test_mode=True)
                flow_up = torch.flip(flow_up_flip, dims=[3])
            else:
                with torch.cuda.amp.autocast(True):
                    flow_up = self.model.forward(left_image, right_image, iters=32, test_mode=True)
            flow_up = padder.unpad(flow_up)
            return flow_up


def disp2depth(disp, baseline, cam):
    assert disp.shape == (1, cam.image_height, cam.image_width)
    focal_length = fov2focal(cam.FoVx, cam.image_width)
    depth = baseline * focal_length / disp
    return depth


def get_occlusion_mask(L2R_disparity, R2L_disparity, occlusion_threshold):
    """
    Calculate the occlusion mask given a pair of disparities.

    Parameters:
    L2R_disparity (np.ndarray): Left-to-right disparity map.
    R2L_disparity (np.ndarray): Right-to-left disparity map.
    occlusion_threshold (int): Threshold on the reprojection error.

    Returns:
    np.ndarray: Binary occlusion mask where 0 indicates occluded pixels and 1 indicates visible pixels.
    """
    height, width = L2R_disparity.shape

    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))

    x_projected = (x_grid - L2R_disparity).astype(np.int32)
    x_projected_clipped = np.clip(x_projected, 0, width - 1)

    x_reprojected = x_projected_clipped + R2L_disparity[y_grid, x_projected_clipped]
    x_reprojected_clipped = np.clip(x_reprojected, 0, width - 1)

    disparity_difference = np.abs(x_grid - x_reprojected_clipped)

    occlusion_mask = (disparity_difference > occlusion_threshold).astype(np.uint8)

    occlusion_mask[(x_projected < 0) | (x_projected >= width)] = 1

    occlusion_mask = occlusion_mask > 0.5

    return ~occlusion_mask


def left_right_check(L2R_disparity, R2L_disparity, occlusion_threshold=3):
    """
    Calculate the occlusion mask for a batch of disparities.

    Parameters:
    L2R_disparity (torch.Tensor): Left-to-right disparity map (batch_size, height, width).
    R2L_disparity (torch.Tensor): Right-to-left disparity map (batch_size, height, width).
    occlusion_threshold (float): Threshold on the reprojection error.

    Returns:
    torch.Tensor: Binary occlusion mask where 0 indicates occluded pixels and 1 indicates visible pixels.
    """
    batch_size, height, width = L2R_disparity.shape

    x_grid = torch.arange(width).view(1, 1, -1).repeat(batch_size, height, 1).to(L2R_disparity.device)

    x_projected = (x_grid - L2R_disparity).long()
    x_projected_clipped = torch.clamp(x_projected, 0, width - 1)

    x_reprojected = x_projected_clipped + R2L_disparity.gather(2, x_projected_clipped)
    x_reprojected_clipped = torch.clamp(x_reprojected, 0, width - 1)

    disparity_difference = torch.abs(x_grid - x_reprojected_clipped)

    occlusion_mask = (disparity_difference > occlusion_threshold).byte()
    occlusion_mask[(x_projected < 0) | (x_projected >= width)] = 1
    return ~occlusion_mask.bool()
