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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch
from utils.feat_utils import FeatExt, load_pair
from utils.general_utils import PILtoTorch
import cv2
import torch.nn.functional as F
import math
from sklearn.decomposition import PCA
from torchvision import transforms
from copy import deepcopy
import imageio
import skimage

def pca_func(feature, n=3):
    device = feature.device
    B, C, H, W = feature.shape
    assert B == 1, B
    pca = PCA(n_components=n)
    feature = feature.permute(0, 2, 3, 1).contiguous().reshape(H*W, C).detach().cpu().numpy()
    feature_map = pca.fit_transform(feature)  # (H*W, 3)
    feature_map = torch.from_numpy(feature_map).to(device)
    feature_map = feature_map.reshape(H, W, n).permute(2, 0, 1).unsqueeze(0)
    return feature_map

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    feat: list = None
    pair: list = None
    mask: np.array = None
    bounds: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def _lookup_name_tokens(value):
    tokens = []
    seen = set()

    def add_token(token):
        if token is None:
            return
        token = str(token).strip()
        if len(token) == 0 or token in seen:
            return
        seen.add(token)
        tokens.append(token)

    add_token(value)
    if isinstance(value, str):
        base_name = os.path.splitext(os.path.basename(value))[0]
        add_token(base_name)
        try:
            add_token(int(base_name))
        except ValueError:
            pass

    return tokens


def select_sparse_view_subset(cam_infos, pair_file, expected_count=None):
    if not os.path.exists(pair_file):
        return None

    subset_pairs = load_pair(pair_file)
    subset_ids = subset_pairs.get('id_list', [])
    if len(subset_ids) == 0:
        return None

    camera_lookup = {}
    for cam in cam_infos:
        for token in _lookup_name_tokens(cam.image_name):
            camera_lookup.setdefault(token, cam)

    selected_cameras = []
    seen_cameras = set()
    for subset_id in subset_ids:
        matched_camera = None
        for token in _lookup_name_tokens(subset_id):
            matched_camera = camera_lookup.get(token)
            if matched_camera is not None:
                break

        if matched_camera is None or id(matched_camera) in seen_cameras:
            continue

        seen_cameras.add(id(matched_camera))
        selected_cameras.append(matched_camera)

    if expected_count is not None and len(selected_cameras) != expected_count:
        return None

    return selected_cameras if len(selected_cameras) > 0 else None

def readColmapCameras(path, cam_extrinsics, cam_intrinsics, images_folder, read_mask, args):
    cam_infos = []

    masks_folder = os.path.join(path, "mask")

    from glob import glob
    def glob_imgs(path):
        imgs = []
        for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
            imgs.extend(glob(os.path.join(path, ext)))
        return imgs

    image_paths = sorted(glob_imgs(images_folder))

    n_images = len(image_paths)

    feats_list = []
    scale_list = [1, 2]
    if args.resolution == 2:
        ori_w, ori_h = 1600, 1200
    else:
        ori_w, ori_h = 768, 576
    for scale in scale_list:
        feat_ext = FeatExt().cuda()
        feat_ext.eval()
        for p in feat_ext.parameters():
            p.requires_grad = False

        size_w = ori_w // scale
        size_h = ori_h // scale

        rgb_2xd = torch.zeros(n_images, 3, size_h, size_w)

        for i in range(n_images):
            image_pil = Image.open(image_paths[i])
            resolution = (int(image_pil.size[0] / scale), int(image_pil.size[1] / scale))
            image = torch.cat([PILtoTorch(im, resolution) for im in image_pil.split()[:3]], dim=0)
            rgb_2xd[i, :, :image.shape[1], :image.shape[2]] = image

        mean = torch.tensor([0.485, 0.456, 0.406]).float()
        std = torch.tensor([0.229, 0.224, 0.225]).float()
        rgb_2xd = (rgb_2xd / 2 + 0.5 - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

        feats = []
        feat_eval_bs = 20
        for start_i in range(0, n_images, feat_eval_bs):
            eval_batch = rgb_2xd[start_i:start_i + feat_eval_bs]
            feat2 = feat_ext(eval_batch.cuda())[2].detach().cpu()
            feats.append(feat2)
        feats = torch.cat(feats, dim=0)
        feats = feats[..., :(ori_h // 2) // scale, :(ori_w // 2) // scale]
        feats_list.append(feats)

    pairs = load_pair(os.path.join(images_folder, "..", "pair.txt"))

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        bounds = np.load(os.path.join(path, 'poses_bounds.npy'))[idx, -2:]

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        mask = None
        if read_mask:
            mask_path = os.path.join(masks_folder, "{:0>3}.png".format(int(image_name)))
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)

        pair = pairs[str(int(image_name))]['pair'][:2]
        # pair = None
        # 

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              feat=[feats[int(image_name)] for feats in feats_list],
                              pair=[int(idx) for idx in pair], mask=mask, bounds=bounds)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readSparseColmapCameras(path, cam_extrinsics, cam_intrinsics, images_folder, read_mask, args):
    cam_infos = []

    masks_folder = os.path.join(path, "mask")

    from glob import glob
    def glob_imgs(path):
        imgs = []
        for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
            imgs.extend(glob(os.path.join(path, ext)))
        return imgs

    image_paths = sorted(glob_imgs(images_folder))

    n_images = len(image_paths)

    if args.n_views == 49:
        pair_path = os.path.join(images_folder, "..", "pair.txt")
    else:
        pair_path = os.path.join(images_folder, "..", str(args.n_views) + "_views", "pair.txt")
    if os.path.exists(pair_path):
        pairs = load_pair(pair_path)
    else:
        pairs = {}
    
    feat_ext = FeatExt().cuda()
    feat_ext.eval()
    for p in feat_ext.parameters():
        p.requires_grad = False
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        # read scene bounds
        pose_file_path = os.path.join(os.path.dirname(images_folder), 'poses_bounds.npy')
        poses_arr = np.load(pose_file_path)
        bds = poses_arr[extr.id-1, -2:]

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        mask = None
        if read_mask:
            mask_path = os.path.join(masks_folder, "{:0>3}.png".format(int(image_name)))
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)

        pair = None
        for pair_key in _lookup_name_tokens(image_name):
            pair_entry = pairs.get(pair_key)
            if pair_entry is None:
                continue
            pair = list(pair_entry['pair'][:2])
            break
        if pair == None:
            feat = None
            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              feat=None,
                              pair=pair, mask=mask, bounds=bds)
        else:
            feats_list = []
            scale_list = [1, 2]
            image_pil = Image.open(image_path)
            if args.resolution == 2:
                ori_w, ori_h = 1600, 1200
                r1=1
            elif args.resolution == 4:
                # ori_w, ori_h = 1600, 1200
                ori_w, ori_h = 3200, 2800
                r1=2
            elif args.resolution == 8:
                ori_w, ori_h = 2400, 2000
                r1=4
            else:
                ori_w, ori_h = image_pil.size
                r1 = 1
            for scale in scale_list:
                size_w = math.ceil((ori_w / scale) / 32) * 32
                size_h = math.ceil((ori_h / scale) / 32) * 32
                rgb_2xd = torch.zeros(1, 3, size_h, size_w)
                resolution_ = (int(image_pil.size[0] // r1 // scale), int(image_pil.size[1] // r1 // scale))
                image_ = torch.cat([PILtoTorch(im, resolution_) for im in image_pil.split()[:3]], dim=0)
                rgb_2xd[0, :, :image_.shape[1], :image_.shape[2]] = image_

                mean = torch.tensor([0.485, 0.456, 0.406]).float()
                std = torch.tensor([0.229, 0.224, 0.225]).float()
                rgb_2xd = (rgb_2xd / 2 + 0.5 - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

                feats = []
                feat_eval_bs = 20
                for start_i in range(0, 1, feat_eval_bs):
                    eval_batch = rgb_2xd[start_i:start_i + feat_eval_bs]
                    feat2 = feat_ext(eval_batch.cuda())[2].detach().cpu()
                    feats.append(feat2)
                feats = torch.cat(feats, dim=0)
                feats = feats[..., :(ori_h // 2) // scale, :(ori_w // 2) // scale]
                feats_list.append(feats)

            cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height,
                                feat=[feats[0] for feats in feats_list],
                                pair=pair, mask=mask, bounds=bds)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def topk_(matrix, K, axis=1):
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        topk_index = np.argpartition(-matrix, K, axis=axis)[0:K, :]
        topk_data = matrix[topk_index, row_index]
        topk_index_sort = np.argsort(-topk_data,axis=axis)
        topk_data_sort = topk_data[topk_index_sort,row_index]
        topk_index_sort = topk_index[0:K,:][topk_index_sort,row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        topk_index = np.argpartition(-matrix, K, axis=axis)[:, 0:K]
        topk_data = matrix[column_index, topk_index]
        topk_index_sort = np.argsort(-topk_data, axis=axis)
        topk_data_sort = topk_data[column_index, topk_index_sort]
        topk_index_sort = topk_index[:,0:K][column_index,topk_index_sort]
    return topk_data_sort

def readColmapSceneInfo(path, images, eval, args, llffhold=8, n_views=3):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    read_mask = True if os.path.exists(os.path.join(path, "mask")) else False
    cam_infos_unsorted = readColmapCameras(path=path, cam_extrinsics=cam_extrinsics,
                                           cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir),
                                           read_mask=read_mask, args=args)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
        exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
        test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
        train_idx = train_idx[:n_views]
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_idx]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_idx]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")

    if eval:
        ply_path = os.path.join(path, str(n_views) + "_views/dense/fused.ply")
    else:
        if os.path.exists(os.path.join(args.reinit_save_path, "stereo_pcd.ply")):
            ply_path = os.path.join(args.reinit_save_path, "stereo_pcd.ply")
            print("create from stereo depth pcd!!!!")
        else:
            ply_path = os.path.join(path, "dense/fused.ply")

    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readIDRCameras(path, args):
    # copy from IDR: https://github.com/lioryariv/idr/

    assert os.path.exists(path), "Data directory is empty"

    from glob import glob
    def glob_imgs(path):
        imgs = []
        for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
            imgs.extend(glob(os.path.join(path, ext)))
        return imgs
    def load_rgb(path):
        img = imageio.imread(path)
        img = skimage.img_as_float32(img)

        # pixel values between [-1,1]
        img -= 0.5
        img *= 2.
        img = img.transpose(2, 0, 1)
        return img

    def load_mask(path):
        alpha = imageio.imread(path, pilmode='F')
        alpha = skimage.img_as_float32(alpha) / 255
        return alpha
    
    # image_dir = '{0}/image'.format(path)
    image_dir = '{0}/images'.format(path)
    image_paths = sorted(glob_imgs(image_dir))
    mask_dir = '{0}/mask'.format(path)
    mask_paths = sorted(glob_imgs(mask_dir))
    cam_file = '{0}/cameras_sphere.npz'.format(path)
    pairs = load_pair(os.path.join(path, "cam4feat", "pair.txt"))
    # pairs = load_pair(os.path.join(path, "pair.txt"))
    n_images = len(image_paths)

    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        # P = world_mat @ scale_mat
        P = world_mat @ scale_mat
        # P = P[:3, :4]
        # intrinsics, pose = load_K_Rt_from_P(None, P)
        pose_all.append(P)


    rgb_images = []
    for i in image_paths:
        rgb = load_rgb(i)
        rgb_images.append(rgb)

    object_masks = []
    for i in mask_paths:
        object_mask = load_mask(i)
        object_masks.append(object_mask[None])
    if len(object_masks) == 0:
        object_masks = [np.ones_like(i[:1]) for i in rgb_images]

    feat_ext = FeatExt().cuda()
    feat_ext.eval()
    for p in feat_ext.parameters():
        p.requires_grad = False

    cam_infos = []
    for i in range(n_images):
        P = pose_all[i]
        K, R, t = cv2.decomposeProjectionMatrix(P[:3, :4])[:3]
        K = K / K[2, 2]
        t = t[:3, :] / t[3:, :]
        T = -R @ t
        T = T[:, 0]
        R = R.T

        image_path = image_paths[i]
        image_name = image_path.split('.')[0].split('/')[-1]
        uid = int(image_name.split('/')[-1])
        image = (rgb_images[i].transpose([1, 2, 0]) * 0.5 + 0.5) * 255

        
        FovY = focal2fov(K[1, 1], image.shape[0])
        FovX = focal2fov(K[0, 0], image.shape[1])
        
        if image.shape[-1] == 4:
            alpha = image[..., 3:] / 255
            object_masks[i] *= alpha.transpose([2, 0, 1])
            image = image[..., :3]
        image = Image.fromarray(np.array(image, dtype=np.byte), "RGB")
        prcppoint = K[:2, 2] / image.size[:2]
        pair = pairs[str(int(image_name))]['pair'][:2]

        feats_list = []
        scale_list = [1, 2]
        if args.resolution == 2:
            ori_w, ori_h = 1600, 1200
            r1=1
        elif args.resolution == 4:
            # ori_w, ori_h = 1600, 1200
            ori_w, ori_h = 3200, 2800
            r1=2
        elif args.resolution == 8:
            ori_w, ori_h = 2400, 2000
            r1=4
        else:
            ori_w, ori_h = 768, 576
            # ori_w, ori_h = 1600, 1200
            r1=1
        for scale in scale_list:
            size_w = ori_w // scale
            size_h = ori_h // scale

            rgb_2xd = torch.zeros(1, 3, size_h, size_w)
            
            image_pil = Image.open(image_path)
            resolution_ = (int(image_pil.size[0] // r1 // scale), int(image_pil.size[1] // r1 // scale))
            image_ = torch.cat([PILtoTorch(im, resolution_) for im in image_pil.split()[:3]], dim=0)
            rgb_2xd[0, :, :image_.shape[1], :image_.shape[2]] = image_

            mean = torch.tensor([0.485, 0.456, 0.406]).float()
            std = torch.tensor([0.229, 0.224, 0.225]).float()
            rgb_2xd = (rgb_2xd / 2 + 0.5 - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

            feats = []
            feat_eval_bs = 20
            for start_i in range(0, 1, feat_eval_bs):
                eval_batch = rgb_2xd[start_i:start_i + feat_eval_bs]
                feat2 = feat_ext(eval_batch.cuda())[2].detach().cpu()
                feats.append(feat2)
            feats = torch.cat(feats, dim=0)
            feats = feats[..., :(ori_h // 2) // scale, :(ori_w // 2) // scale]
            feats = F.interpolate(feats, scale_factor=2, mode='bilinear', align_corners=False)
            feats_list.append(feats)
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                              feat=[feats[0] for feats in feats_list],
                              pair=[int(idx) for idx in pair], mask=object_masks[i], bounds=None)
        cam_infos.append(cam_info)
    return cam_infos, scale_mats

def readIDRSceneInfo(path, eval, args, testskip=8):
    cam_infos, scale_mats = readIDRCameras(path, args)

    if eval:
        # test_cams = [i for i in range(len(cam_infos)) if i % testskip == 0]
        # test split following NeuS2
        test_cams = [8, 13, 16, 21, 26, 31, 34]
        if len(cam_infos) > 56:
            test_cams.append(56)

        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx not in test_cams]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_cams]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    
    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_name = os.path.basename(path)
    ply_path = os.path.join(path, "pcd", scene_name + ".ply")
    if os.path.exists(ply_path):
        pcd = fetchPly(ply_path)
        print(f"Featching points3d.ply...")
    else:
        num_pts = 100_0000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        rand_scale = 1.2
        normal = np.random.random((num_pts, 3)) - 0.5
        normal /= np.linalg.norm(normal, 2, 1, True)
        xyz = normal * 0.5 #- rand_scale / 2

        rand_scale *= 2
        xyz = np.random.random((num_pts, 3)) * rand_scale - rand_scale / 2

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=normal)

    scene_info = SceneInfo(point_cloud=pcd,
                        train_cameras=train_cam_infos,
                        test_cameras=test_cam_infos,
                        nerf_normalization=nerf_normalization,
                        ply_path=ply_path)
    
    return scene_info, scale_mats

def readMipNerfSceneInfo(path, images, eval, args, n_views=0, llffhold=8, rand_pcd=False):
    if n_views <= 0:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
    elif rand_pcd:
        print('Init random point cloud.')
        ply_path = os.path.join(path, "sparse/0/points3D_random.ply")
        bin_path = os.path.join(path, "sparse/0/points3D_random.bin")
        txt_path = os.path.join(path, "sparse/0/points3D_random.txt")

        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)

        pcd_shape = (topk_(xyz, 1, 0)[-1] + topk_(-xyz, 1, 0)[-1])
        num_pts = int(pcd_shape.max() * 50)
        xyz = np.random.random((num_pts, 3)) * pcd_shape * 1.3 - topk_(-xyz, 20, 0)[-1]
        print(pcd_shape)
        print(f"Generating random point cloud ({num_pts})...")

        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        ply_path = os.path.join(path, str(n_views) + "_views/dense/fused.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")

    try:
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    

    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        os.makedirs(os.path.dirname(ply_path), exist_ok=True)
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None


    reading_dir = "images" if images == None else images
    read_mask = True if os.path.exists(os.path.join(path, "mask")) else False
    cam_infos_unsorted = readSparseColmapCameras(path=path, cam_extrinsics=cam_extrinsics,
                                           cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir),
                                           read_mask=read_mask, args=args)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    if n_views > 0:
        sparse_pair_file = os.path.join(path, str(n_views) + "_views", "pair.txt")
        sparse_subset = select_sparse_view_subset(train_cam_infos, sparse_pair_file, expected_count=n_views)
        if sparse_subset is not None:
            train_cam_infos = sparse_subset
        else:
            idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views)
            idx_sub = [round(i) for i in idx_sub]
            train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx in idx_sub]
            assert len(train_cam_infos) == n_views

    nerf_normalization = getNerfppNorm(train_cam_infos)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readSparseColmapSceneInfo(path, images, eval, args, llffhold=8, n_views=3, rand_pcd=False):
    if rand_pcd:
        print('Init random point cloud.')
        ply_path = os.path.join(path, "sparse/0/points3D_random.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")

        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        print(xyz.max(0), xyz.min(0))
        pcd_shape = (topk_(xyz, 100, 0)[-1] + topk_(-xyz, 100, 0)[-1])
        num_pts = 10_00
        xyz = np.random.random((num_pts, 3)) * pcd_shape * 1.3 - topk_(-xyz, 100, 0)[-1] # - 0.15 * pcd_shape
        print(pcd_shape)
        print(f"Generating random point cloud ({num_pts})...")
        shs = np.random.random((num_pts, 3)) / 255.0
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        if n_views== 49 :
            ply_path = os.path.join(path, "sparse/0/points3D.ply")
        else:
            ply_path = os.path.join(path, str(n_views) + "_views/dense/fused.ply")
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    read_mask = True if os.path.exists(os.path.join(path, "mask")) else False
    cam_infos_unsorted = readSparseColmapCameras(path=path, cam_extrinsics=cam_extrinsics,
                                           cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir),
                                           read_mask=read_mask, args=args)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
        exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
        test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
        train_idx = train_idx[:n_views]
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_idx]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_idx]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")

    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Sparse": readSparseColmapSceneInfo,
    "MipNerf": readMipNerfSceneInfo,
    "IDR": readIDRSceneInfo
}
