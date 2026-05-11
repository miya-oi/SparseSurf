import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCANS = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]


def parse_args():
    parser = argparse.ArgumentParser(description="Train, extract mesh, and evaluate DTU mesh scenes.")
    parser.add_argument("source_path", help="Single DTU scan path or a root that contains scanXX folders.")
    parser.add_argument("output_path", help="Output scene directory or output root.")
    parser.add_argument("gpu_id", help="CUDA device id.")
    parser.add_argument(
        "dtu_gt_root",
        help="DTU evaluation root, e.g. /path/to/dtu_2dgs/DTU/SampleSet/MVS_Data.",
    )
    parser.add_argument(
        "--mask_root",
        default=None,
        help="Optional alternate root for DTU masks. May be a dataset root containing scanXX/mask or a direct scene root.",
    )
    parser.add_argument("--scans", nargs="+", type=int, default=DEFAULT_SCANS)
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--total_virtual_num", type=int, default=240)
    parser.add_argument("--sdf_trunc_mul", type=float, default=4.0)
    parser.add_argument("--foundation_stereo_ckpt", default=os.environ.get("FOUNDATION_STEREO_CKPT", ""))
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    return parser.parse_args()


def run_command(command, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=env)


def _list_valid_pngs(directory: Path):
    return sorted(
        path
        for path in directory.glob("*.png")
        if path.is_file() and not path.name.startswith("._")
    )


def ensure_scene_masks(mask_root: Path, scan_id: int):
    mask_dir = mask_root / f"scan{scan_id}" / "mask"
    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Missing mask directory for scan{scan_id}: {mask_dir}\n"
            "If your downloaded DTU masks live elsewhere, pass --mask_root explicitly."
        )

    valid_masks = _list_valid_pngs(mask_dir)
    if not valid_masks:
        raise FileNotFoundError(
            f"No usable PNG masks were found in {mask_dir}.\n"
            "Files like '._000.png' are macOS AppleDouble sidecars, not real masks.\n"
            "Please provide a clean mask root via --mask_root."
        )


def _is_valid_eval_scene(scene_dir: Path):
    return (
        scene_dir.is_dir()
        and (scene_dir / "images").is_dir()
        and (scene_dir / "cameras.npz").is_file()
        and (scene_dir / "mask").is_dir()
        and bool(_list_valid_pngs(scene_dir / "mask"))
    )


def _mask_root_candidates(explicit_mask_root, source_path, dtu_gt_root):
    if explicit_mask_root is not None:
        yield Path(explicit_mask_root)
        return

    source_path = Path(source_path)
    if source_path.name.startswith("scan"):
        yield source_path.parent
    else:
        yield source_path

    dtu_gt_root = Path(dtu_gt_root)
    for candidate in (dtu_gt_root, dtu_gt_root.parent, dtu_gt_root.parent.parent):
        if str(candidate) != ".":
            yield candidate


def resolve_mask_root(explicit_mask_root, source_path, dtu_gt_root, scan_id):
    seen = set()
    candidates = list(_mask_root_candidates(explicit_mask_root, source_path, dtu_gt_root))

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)

        mask_dir = candidate / f"scan{scan_id}" / "mask"
        if not mask_dir.is_dir():
            continue
        if _list_valid_pngs(mask_dir):
            return candidate

    if explicit_mask_root is not None:
        ensure_scene_masks(Path(explicit_mask_root), scan_id)

    for candidate in candidates:
        mask_dir = candidate / f"scan{scan_id}" / "mask"
        if mask_dir.is_dir():
            ensure_scene_masks(candidate, scan_id)

    raise FileNotFoundError(
        "Unable to find usable DTU masks automatically.\n"
        f"Tried source/eval-adjacent roots for scan{scan_id}, including {Path(source_path)} and {Path(dtu_gt_root)}.\n"
        "Pass --mask_root /path/to/dtu_2dgs/DTU if your masks come from the 2DGS-processed DTU data."
    )


def resolve_eval_scene_dir(mask_root: Path, source_path, scan_id):
    source_path = Path(source_path).resolve()
    eval_candidates = []

    for candidate in (mask_root / f"scan{scan_id}", mask_root):
        resolved_candidate = candidate.resolve()
        if resolved_candidate not in eval_candidates:
            eval_candidates.append(resolved_candidate)

    for candidate in eval_candidates:
        if _is_valid_eval_scene(candidate):
            return candidate

    if _is_valid_eval_scene(source_path):
        return source_path

    raise FileNotFoundError(
        f"Unable to find a valid DTU eval scene for scan{scan_id}.\n"
        f"Tried {mask_root / f'scan{scan_id}'} and {mask_root}, then fell back to {source_path}.\n"
        "A valid eval scene must contain images/, cameras.npz, and mask/ with real PNG masks."
    )


def build_scene_entries(source_path, output_path, scans):
    source = Path(source_path)
    output = Path(output_path)
    if source.name.startswith("scan"):
        return [(int(source.name[4:]), source, output)]
    return [(scan, source / f"scan{scan}", output / f"scan{scan}") for scan in scans]


def main():
    args = parse_args()
    scene_entries = build_scene_entries(args.source_path, args.output_path, args.scans)
    mask_roots = {}
    eval_scene_dirs = {}

    if not args.skip_eval:
        for scan_id, source_path, _ in scene_entries:
            mask_roots[scan_id] = resolve_mask_root(args.mask_root, source_path, args.dtu_gt_root, scan_id)
            eval_scene_dirs[scan_id] = resolve_eval_scene_dir(mask_roots[scan_id], source_path, scan_id)

    for scan_id, source_path, output_path in scene_entries:
        output_path.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable,
            str(REPO_ROOT / "train.py"),
            "-s",
            str(source_path),
            "-m",
            str(output_path),
            "-r",
            str(args.resolution),
            "--iterations",
            str(args.iterations),
            "--total_virtual_num",
            str(args.total_virtual_num),
        ]
        if args.foundation_stereo_ckpt:
            train_cmd.extend(["--foundation_stereo_ckpt", args.foundation_stereo_ckpt])

        mesh_cmd = [
            sys.executable,
            str(REPO_ROOT / "extract_mesh.py"),
            "-s",
            str(source_path),
            "-m",
            str(output_path),
            "-r",
            str(args.resolution),
            "--iteration",
            str(args.iterations),
            "--sdf_trunc_mul",
            str(args.sdf_trunc_mul),
        ]

        eval_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "eval_dtu" / "evaluate_single_scene.py"),
            "--input_mesh",
            str(output_path / "mesh" / "tsdf_fusion_post.ply"),
            "--scan_id",
            str(scan_id),
            "--output_dir",
            str(output_path / "result"),
            "--scene_dir",
            str(eval_scene_dirs[scan_id]),
            "--DTU",
            str(args.dtu_gt_root),
        ]

        if not args.skip_train:
            run_command(train_cmd, args.gpu_id)
        if not args.skip_mesh:
            run_command(mesh_cmd, args.gpu_id)
        if not args.skip_eval:
            run_command(eval_cmd, args.gpu_id)


if __name__ == "__main__":
    main()
