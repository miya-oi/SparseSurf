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
from torch import nn
import numpy as np
import math
from PIL import Image
import torch.nn.functional as F
from utils.general_utils import PILtoTorch
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal

def process_image(image_path, resolution, ncc_scale):
    image = Image.open(image_path)
    if len(image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(image.split()[3], resolution)
        gt_image = resized_image_rgb
        if ncc_scale != 1.0:
            ncc_resolution = (int(resolution[0]/ncc_scale), int(resolution[1]/ncc_scale))
            resized_image_rgb = torch.cat([PILtoTorch(im, ncc_resolution) for im in image.split()[:3]], dim=0)
    else:
        resized_image_rgb = PILtoTorch(image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb
        if ncc_scale != 1.0:
            ncc_resolution = (int(resolution[0]/ncc_scale), int(resolution[1]/ncc_scale))
            resized_image_rgb = PILtoTorch(image, ncc_resolution)
    gray_image = (0.299 * resized_image_rgb[0] + 0.587 * resized_image_rgb[1] + 0.114 * resized_image_rgb[2])[None]
    return gt_image, gray_image, loaded_mask

def process_right_image(image, resolution, ncc_scale):
    resized_image_rgb = image
    if ncc_scale != 1.0:
        ncc_resolution = (int(resolution[1]/ncc_scale), int(resolution[0]/ncc_scale))
        resized_image_rgb = F.interpolate(image.unsqueeze(0), size=ncc_resolution, mode='bilinear', align_corners=False)
        resized_image_rgb = resized_image_rgb.squeeze(0)
        # resized_image_rgb = PILtoTorch(image, ncc_resolution)
    # gray_image = (0.299 * resized_image_rgb[0] + 0.587 * resized_image_rgb[1] + 0.114 * resized_image_rgb[2])[None]
    gray_image = (0.299 * resized_image_rgb[0] + 0.587 * resized_image_rgb[1] + 0.114 * resized_image_rgb[2])[None]
    return gray_image

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, image_path, uid, feats, pair, use_mask=False,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 preload_img=True, is_virtual=False, warped_mask=None, warped_gt_feat=None, bounds=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.pair = pair
        self.image_path = image_path
        self.is_virtual = is_virtual
        self.warped_gt_feat = warped_gt_feat
        self.bounds = bounds

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.Fx = fov2focal(FoVx, self.image_width)
        self.Fy = fov2focal(FoVy, self.image_height)
        self.Cx = 0.5 * self.image_width
        self.Cy = 0.5 * self.image_height
        self.K = torch.tensor([
            [self.Fx, 0.0, self.Cx],
            [0.0, self.Fy, self.Cy],
            [0.0, 0.0, 1.0]
        ]).to(self.data_device)

        self.last_stereo_depth = None
        self.last_stereo_depth_mask = None
        self.last_stereo_iter = -1
        self.last_right_transforms = None
        self.last_right_depth = None
        self.last_right_image = None
        self.last_right_gray_image = None
        self.last_right_feat = None
        self.resolution = (self.image_width, self.image_height)

        if gt_alpha_mask is not None:
            if use_mask:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None

        if feats is None:
            self.feat = None
        else:
            # print(f"feats.shape is {feats.shape}")
            # print(f"image.shape is {image.shape}")
            self.feat = [
                feats[0][:, :math.ceil(image.shape[1]), :math.ceil(image.shape[2])].to(self.data_device),
                feats[1][:, :math.ceil(image.shape[1] / 2), :math.ceil(image.shape[2] / 2)].to(self.data_device)
            ]
            # print(f"feat[0].shape is {self.feat[0].shape}")
            # print(f"feat[1].shape is {self.feat[1].shape}")

        self.w2c = torch.tensor(getWorld2View2(R, T, trans, scale)).cuda()
        self.c2w = self.w2c.inverse().cuda()
        fx = self.image_width / (2 * math.tan(self.FoVx / 2.))
        fy = self.image_height / (2 * math.tan(self.FoVy / 2.))
        self.intrinsic = torch.tensor(
            [[fx, 0., self.image_width/2., 0.],
            [0., fy, self.image_height/2., 0.],
            [0., 0., 1.0, 0.],
            [0., 0., 0., 1.0]]
        ).float().cuda()

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.ncc_scale = 1.0
        self.preload_img = preload_img

        if self.preload_img:
            gt_image, gray_image, loaded_mask = process_image(self.image_path, self.resolution, self.ncc_scale)
            self.original_image = gt_image.to(self.data_device)
            self.original_image_gray = gray_image.to(self.data_device)
            self.mask = loaded_mask

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.right_cam = None
        self.warped_gt_image = None
        self.warped_mask = warped_mask

    def get_image(self):
        if self.preload_img:
            return self.original_image.cuda(), self.original_image_gray.cuda()
        else:
            gt_image, gray_image, _ = process_image(self.image_path, self.resolution, self.ncc_scale)
            return gt_image.cuda(), gray_image.cuda()

    def get_k(self, scale=1.0):
        K = torch.tensor([[self.Fx / scale, 0, self.Cx / scale],
                        [0, self.Fy / scale, self.Cy / scale],
                        [0, 0, 1]], device='cuda')
        return K
    
    def get_inv_k(self, scale=1.0):
        K_T = torch.tensor([[scale/self.Fx, 0, -self.Cx/self.Fx],
                            [0, scale/self.Fy, -self.Cy/self.Fy],
                            [0, 0, 1]], device='cuda')
        return K_T
    
    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor([[self.Fx/scale, 0, self.Cx/scale], [0, self.Fy/scale, self.Cy/scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix
    
    def set_last_stereo_depth(self, depth, iter):
        self.last_stereo_depth = depth.to(self.data_device)
        self.last_stereo_iter = iter

    def set_last_stereo_depth_mask(self, mask):
        self.last_stereo_depth_mask = mask.to(self.data_device)
        # self.last_stereo_iter = iter

    def set_last_right_transforms(self, transforms):
        self.last_right_transforms = transforms.to(self.data_device)

    def set_last_right_depth(self, depth):
        self.last_right_depth = depth.to(self.data_device)

    def set_last_right_image(self, image):
        self.last_right_image = image.to(self.data_device)
        gray_image = process_right_image(image, self.resolution, self.ncc_scale)
        self.last_right_gray_image = gray_image.to(self.data_device)
    
    def set_last_right_feat(self, feat):
        self.last_right_feat = feat.to(self.data_device)

    def get_last_stereo_depth(self):
        return self.last_stereo_depth.cuda()

    def get_last_stereo_depth_mask(self):
        return self.last_stereo_depth_mask.cuda()

    def get_last_right_transforms(self):
        return self.last_right_transforms.cuda()

    def get_last_right_depth(self):
        return self.last_right_depth.cuda()
    
    def get_last_right_image(self):
        return self.last_right_image.cuda()

    def get_last_right_gray_image(self):
        return self.last_right_gray_image.cuda()
    
    def get_last_right_feat(self):
        return self.last_right_feat.cuda()

    def get_stereo_pair(self, baseline=0.3):
        tr = np.array([-baseline,0.,0.])
        return Camera(colmap_id=self.colmap_id, R=self.R, T=self.T+tr, 
                FoVx=self.FoVx, FoVy=self.FoVy, 
                image=self.original_image, gt_alpha_mask=None,
                image_name=f'{self.image_name}_right', image_path=None, uid=None, 
                feats=None, pair=None,
                data_device=self.data_device, preload_img=False)

    def get_stereo_pair_(self, baseline=0.3):
        tr = np.array([-baseline,0.,0.])
        return Camera(colmap_id=self.colmap_id, R=self.R, T=self.T-tr, 
                FoVx=self.FoVx, FoVy=self.FoVy, 
                image=self.original_image, gt_alpha_mask=None,
                image_name=f'{self.image_name}_right', image_path=None, uid=None, 
                feats=None, pair=None,
                data_device=self.data_device, preload_img=False)
    
    def get_rays(self, scale=1.0):
        W, H = int(self.image_width/scale), int(self.image_height/scale)
        ix, iy = torch.meshgrid(
            torch.arange(W, device="cuda"), torch.arange(H, device="cuda"), indexing='xy')
        rays_d = torch.stack(
                    [(ix-self.Cx/scale) / self.Fx * scale,
                    (iy-self.Cy/scale) / self.Fy * scale,
                    torch.ones_like(ix)], -1).float()
        return rays_d

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
