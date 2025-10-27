from pathlib import Path
import shutil

from tqdm import tqdm
import numpy as np


def quat_to_matrix(qw, qx, qy, qz, tx, ty, tz):
    R = np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
    ])

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = [tx, ty, tz]
    return pose


def convert_eth3d_format(scene_path):
    scene_path = Path(scene_path)

    img_subdir = scene_path / 'images/dslr_images_undistorted'
    if img_subdir.exists():
        for img_file in img_subdir.glob('*.JPG'):
            shutil.move(str(img_file), str(scene_path / 'images'))
        img_subdir.rmdir()

    images_file = scene_path / 'dslr_calibration_undistorted/images.txt'
    poses_dir = scene_path / 'poses'
    poses_dir.mkdir(exist_ok=True)

    with open(images_file) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            img_name = Path(parts[-1]).stem

            qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
            pose = quat_to_matrix(qw, qx, qy, qz, tx, ty, tz)

            np.savetxt(poses_dir / f'{img_name}.txt', pose)


scenes = ['courtyard', 'delivery_area', 'electro', 'facade', 'kicker', 'meadow',
          'office', 'pipes', 'playground', 'relief', 'relief_2', 'terrace', 'terrains']

for scene in tqdm(scenes):
    convert_eth3d_format(f'SfM_dataset/eth3d_triangulation_dataset/{scene}')
