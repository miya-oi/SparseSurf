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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.pose_utils import generate_random_poses_360, generate_random_poses
from scene.cameras import Camera
import numpy as np
import torch
from tqdm import tqdm

class Scene:

    gaussians : GaussianModel
    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=False, resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.args = args
        self.resolution_scales = resolution_scales
        # self.stage = stage

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
        self.dataset_format = None
        self.train_cameras = {}
        self.test_cameras = {}
        self.scale_mats = None
        self.virtual_cameras = {} # unseen view virtual cams
        if args.source_path.find('mipnerf360') != -1:
            print("############ load mipnerf360 ############")
            scene_info = sceneLoadTypeCallbacks["MipNerf"](args.source_path, args.images, args.eval, args, n_views=args.n_views)
            self.dataset_format = 'M360'
        elif os.path.exists(os.path.join(args.source_path, "3_views")):
            print("############ load NVS DTU ############")
            scene_info = sceneLoadTypeCallbacks["Sparse"](args.source_path, args.images, args.eval, args, n_views=args.n_views)
            self.dataset_format = 'DTU'
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            print("############ load Mesh DTU ############")
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args, n_views=args.n_views)
            self.dataset_format = 'DTU'
        elif os.path.exists(os.path.join(args.source_path, "cameras_sphere.npz")):
            # BlendedMVS dataset format
            print("Found camera.npz file, assuming IDR data format!")
            scene_info, scale_mats = sceneLoadTypeCallbacks["IDR"](args.source_path, args.eval, args)
            self.scale_mats = scale_mats
            self.dataset_format = 'DTU'
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print("Generating Virtual Cameras, num: ", args.total_virtual_num)
            # generate pseudo cameras
            pseudo_cams = []
            has_forward_facing_bounds = self._has_valid_view_bounds(self.train_cameras[resolution_scale])
            if self.dataset_format == 'M360' or not has_forward_facing_bounds:
                pseudo_poses = generate_random_poses_360(self.train_cameras[resolution_scale], n_poses=args.total_virtual_num)
            elif self.dataset_format == 'DTU':
                pseudo_poses = generate_random_poses(self.train_cameras[resolution_scale], n_poses=args.total_virtual_num)
            view = self.train_cameras[resolution_scale][0]
            idx=0
            for pose in pseudo_poses:
                pseudo_cams.append(Camera(colmap_id=None, R=pose[:3, :3].T, T=pose[:3, 3], 
                                    FoVx=view.FoVx, FoVy=view.FoVy, 
                                    image=view.original_image.cpu(), gt_alpha_mask=None,
                                    image_name='virtual_'+str(idx), uid=idx, data_device='cpu', 
                                    is_virtual=True, warped_mask=None, 
                                    image_path=None, feats=None, pair=None, preload_img=False, warped_gt_feat=None))
                idx+=1   
            self.virtual_cameras[resolution_scale] = pseudo_cams
            torch.cuda.empty_cache()


        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTrainCamerasByIdx(self, idx, scale=1.0):
        cameras = self.train_cameras[scale]
        return [cameras[i] for i in idx]

    def _has_valid_view_bounds(self, views):
        for view in views:
            bounds = getattr(view, "bounds", None)
            if bounds is None:
                return False

            bounds_array = np.asarray(bounds).reshape(-1)
            if bounds_array.size < 2:
                return False

            near, far = bounds_array[:2]
            if not np.isfinite(near) or not np.isfinite(far) or far <= near:
                return False

        return True

    def _lookup_tokens(self, value):
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
            add_token(os.path.splitext(os.path.basename(value))[0])
            try:
                add_token(int(value))
            except ValueError:
                pass

        return tokens

    def _camera_lookup_tokens(self, camera, camera_idx=None):
        tokens = []
        if camera_idx is not None:
            tokens.extend(self._lookup_tokens(camera_idx))
        tokens.extend(self._lookup_tokens(getattr(camera, "uid", None)))
        tokens.extend(self._lookup_tokens(getattr(camera, "colmap_id", None)))
        tokens.extend(self._lookup_tokens(getattr(camera, "image_name", None)))

        unique_tokens = []
        seen = set()
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        return unique_tokens

    def _resolve_train_camera(self, selector, cameras):
        if isinstance(selector, (int, np.integer)) and 0 <= int(selector) < len(cameras):
            return cameras[int(selector)]

        lookup = {}
        for camera_idx, camera in enumerate(cameras):
            for token in self._camera_lookup_tokens(camera, camera_idx):
                lookup.setdefault(token, camera)

        for token in self._lookup_tokens(selector):
            if token in lookup:
                return lookup[token]

        raise KeyError(f"Unable to resolve training camera selector: {selector}")

    def _fallback_train_sources(self, ref_camera, cameras, count):
        if len(cameras) <= 1:
            return []

        ref_center = ref_camera.c2w[:3, 3]
        neighbors = []
        for camera in cameras:
            if camera is ref_camera:
                continue
            distance = torch.norm(camera.c2w[:3, 3] - ref_center).item()
            neighbors.append((distance, camera))

        neighbors.sort(key=lambda x: x[0])
        return [camera for _, camera in neighbors[:count]]

    def getTrainCamerasSource(self, cam_img_name, scale=1.0):
        cameras = self.train_cameras[scale]
        ref_camera = self._resolve_train_camera(cam_img_name, cameras)

        desired_count = max(len(getattr(ref_camera, "pair", []) or []), 2)
        desired_count = min(desired_count, max(len(cameras) - 1, 0))

        camera_lookup = {}
        for camera_idx, camera in enumerate(cameras):
            for token in self._camera_lookup_tokens(camera, camera_idx):
                camera_lookup.setdefault(token, camera)

        source_cameras = []
        seen_cameras = {id(ref_camera)}
        for pair_item in getattr(ref_camera, "pair", []) or []:
            matched_camera = None
            for token in self._lookup_tokens(pair_item):
                matched_camera = camera_lookup.get(token)
                if matched_camera is not None:
                    break
            if matched_camera is None or id(matched_camera) in seen_cameras:
                continue
            seen_cameras.add(id(matched_camera))
            source_cameras.append(matched_camera)

        if len(source_cameras) < desired_count:
            for fallback_camera in self._fallback_train_sources(ref_camera, cameras, desired_count):
                if id(fallback_camera) in seen_cameras:
                    continue
                seen_cameras.add(id(fallback_camera))
                source_cameras.append(fallback_camera)
                if len(source_cameras) >= desired_count:
                    break

        return source_cameras

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getVirtualCameras(self, scale=1.0):
        return self.virtual_cameras[scale]
