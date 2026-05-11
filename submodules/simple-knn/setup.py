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

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
