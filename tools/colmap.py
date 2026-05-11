import argparse
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import numpy as np


IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid

def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded


TRAIN_INDICES = [25, 22, 28, 40, 44, 48, 0, 8, 13]


def run_command(command, cwd):
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=cwd)


def parse_images_txt(images_txt_path):
    images = {}
    with open(images_txt_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_name = elems[9]
                fid.readline()
                images[image_name] = elems[1:]
    return images


def select_train_images(images, n_views):
    img_list = sorted(images.keys())
    if n_views <= 0:
        return img_list

    selected_indices = set(TRAIN_INDICES[:n_views])
    return [img_name for idx, img_name in enumerate(img_list) if idx in selected_indices]


def pipeline(scene, base_path, n_views):
    scene_root = Path(base_path) / scene
    sparse_dir = scene_root / "sparse" / "0"
    image_dir = scene_root / "images"
    view_dir = scene_root / f"{n_views}_views"

    if not scene_root.is_dir():
        raise FileNotFoundError(f"Scene directory does not exist: {scene_root}")
    if not sparse_dir.is_dir():
        raise FileNotFoundError(f"Missing sparse reconstruction: {sparse_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {image_dir}")

    if view_dir.exists():
        shutil.rmtree(view_dir)
    (view_dir / "created").mkdir(parents=True)
    (view_dir / "triangulated").mkdir()
    (view_dir / "images").mkdir()

    run_command(
        [
            "colmap",
            "model_converter",
            "--input_path",
            str(sparse_dir),
            "--output_path",
            str(sparse_dir),
            "--output_type",
            "TXT",
        ],
        cwd=scene_root,
    )

    images = parse_images_txt(sparse_dir / "images.txt")
    train_img_list = select_train_images(images, n_views)
    print(f"{scene}: selected {len(train_img_list)} training images")

    for img_name in train_img_list:
        shutil.copy2(image_dir / img_name, view_dir / "images" / img_name)

    shutil.copy2(sparse_dir / "cameras.txt", view_dir / "created" / "cameras.txt")
    (view_dir / "created" / "points3D.txt").write_text("")

    run_command(
        [
            "colmap",
            "feature_extractor",
            "--database_path",
            "database.db",
            "--image_path",
            "images",
            "--SiftExtraction.max_image_size",
            "4032",
            "--SiftExtraction.max_num_features",
            "32768",
            "--SiftExtraction.estimate_affine_shape",
            "1",
            "--SiftExtraction.domain_size_pooling",
            "1",
        ],
        cwd=view_dir,
    )
    run_command(
        [
            "colmap",
            "exhaustive_matcher",
            "--database_path",
            "database.db",
            "--SiftMatching.guided_matching",
            "1",
            "--SiftMatching.max_num_matches",
            "32768",
        ],
        cwd=view_dir,
    )

    db = COLMAPDatabase.connect(str(view_dir / "database.db"))
    db_images = db.execute("SELECT * FROM images")
    img_rank = [db_image[1] for db_image in db_images]
    with open(view_dir / "created" / "images.txt", "w") as fid:
        for idx, img_name in enumerate(img_rank):
            data = [str(1 + idx)] + [' ' + item for item in images[os.path.basename(img_name)]] + ['\n\n']
            fid.writelines(data)
    db.close()

    run_command(
        [
            "colmap",
            "point_triangulator",
            "--database_path",
            "database.db",
            "--image_path",
            "images",
            "--input_path",
            "created",
            "--output_path",
            "triangulated",
            "--Mapper.ba_local_max_num_iterations",
            "40",
            "--Mapper.ba_local_max_refinements",
            "3",
            "--Mapper.ba_global_max_num_iterations",
            "100",
        ],
        cwd=view_dir,
    )
    run_command(
        [
            "colmap",
            "model_converter",
            "--input_path",
            "triangulated",
            "--output_path",
            "triangulated",
            "--output_type",
            "TXT",
        ],
        cwd=view_dir,
    )
    run_command(
        [
            "colmap",
            "image_undistorter",
            "--image_path",
            "images",
            "--input_path",
            "triangulated",
            "--output_path",
            "dense",
        ],
        cwd=view_dir,
    )
    run_command(["colmap", "patch_match_stereo", "--workspace_path", "dense"], cwd=view_dir)
    run_command(
        ["colmap", "stereo_fusion", "--workspace_path", "dense", "--output_path", "dense/fused.ply"],
        cwd=view_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Create sparse-view COLMAP subsets and dense reconstructions.")
    parser.add_argument("--base_path", required=True, help="Dataset root containing the target scenes.")
    parser.add_argument("--scenes", nargs="+", required=True, help="Scene names under base_path.")
    parser.add_argument("--n_views", type=int, default=3, help="Number of training views to keep. Use negative to keep all.")
    args = parser.parse_args()

    for scene in args.scenes:
        pipeline(scene, base_path=args.base_path, n_views=args.n_views)


if __name__ == "__main__":
    main()
