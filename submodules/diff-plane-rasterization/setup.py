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
from pathlib import Path

import torch
from setuptools import setup


def configure_cuda_toolkit():
    torch_cuda = torch.version.cuda
    if not torch_cuda:
        return

    requested_home = os.environ.get("CUDA_HOME")
    candidate_paths = []
    if requested_home:
        candidate_paths.append(Path(requested_home))
    candidate_paths.extend(
        [
            Path(f"/usr/local/cuda-{torch_cuda}"),
            Path(f"/opt/cuda-{torch_cuda}"),
        ]
    )

    for candidate in candidate_paths:
        if not candidate.is_dir():
            continue
        resolved_name = candidate.resolve().name
        if torch_cuda not in resolved_name and candidate.name != f"cuda-{torch_cuda}":
            continue
        os.environ["CUDA_HOME"] = str(candidate)
        os.environ["CUDA_PATH"] = str(candidate)
        nvcc_path = candidate / "bin" / "nvcc"
        if nvcc_path.is_file():
            os.environ["CUDACXX"] = str(nvcc_path)
            os.environ["PATH"] = f"{candidate / 'bin'}:{os.environ.get('PATH', '')}"
        break


configure_cuda_toolkit()

from torch.utils.cpp_extension import CUDAExtension, BuildExtension

os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_plane_rasterization",
    packages=['diff_plane_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_plane_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
