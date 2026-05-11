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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch

WARNED = False

# def write_depth(path, depth, grayscale, bits=1):
#     """Write depth map to png file.

#     Args:
#         path (str): filepath without extension
#         depth (array): depth
#         grayscale (bool): use a grayscale colormap?
#     """
#     if not grayscale:
#         bits = 1

#     if not np.isfinite(depth).all():
#         depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
#         print("WARNING: Non-finite depth values present")

#     depth_min = depth.min()
#     depth_max = depth.max()

#     max_val = (2**(8*bits))-1

#     if depth_max - depth_min > np.finfo("float").eps:
#         out = max_val * (depth - depth_min) / (depth_max - depth_min)
#     else:
#         out = np.zeros(depth.shape, dtype=depth.dtype)

#     if not grayscale:
#         out = cv2.applyColorMap(np.uint8(out), cv2.COLORMAP_INFERNO)

#     if bits == 1:
#         cv2.imwrite(path + ".png", out.astype("uint8"))
#     elif bits == 2:
#         cv2.imwrite(path + ".png", out.astype("uint16"))

#     return

def resize_image(img, factor, mode='bilinear'):
    if factor == 1:
        return img
    is_np = type(img) == np.ndarray
    if is_np:
        resize = torch.from_numpy(img)
    else:
        resize = img.clone()
    dtype = resize.dtype

    if type(factor) == int:
        resize = torch.nn.functional.interpolate(resize[None].to(torch.float32), scale_factor=1/factor, mode=mode)[0].to(dtype)
    elif len(factor) == 2:
        resize = torch.nn.functional.interpolate(resize[None].to(torch.float32), size=factor, mode=mode)[0].to(dtype)

    if is_np:
        resize = resize.numpy()
    return resize

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if len(cam_info.image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution) # (1, H, W)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        if cam_info.mask is not None:
            loaded_mask = resize_image(cam_info.mask, [resolution[1], resolution[0]])
            loaded_mask = torch.from_numpy(loaded_mask > 0).float()
        gt_image = resized_image_rgb

    # image_path = None
    # if cam_info.image_path is not None:
    #     image_path = cam_info.image_path

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, image_path=cam_info.image_path, uid=id,
                  feats=cam_info.feat, pair=cam_info.pair,
                  use_mask=args.use_mask, data_device=args.data_device, bounds=cam_info.bounds)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
