#!/usr/bin/env python3
"""
Complete ETH3D dataset processor for DetectorFreeSfM triangulation evaluation
"""
from pathlib import Path
import numpy as np
import shutil

# ============ 配置路径 ============
INPUT_DIR = Path("/home/ubuntu/code/DenseSfM/data/eth3d_high_res_multi_view")
OUTPUT_DIR = Path("/home/ubuntu/code/DetectorFreeSfM/SfM_dataset/eth3d_triangulation_dataset")

SCENES = ["kicker", "office", "delivery_area", "facade", "meadow", "relief",
          "electro", "terrace", "terrains", "courtyard", "relief_2", "pipes", "playground"]


# ============ 辅助函数 ============
def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix"""
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
    ])


def parse_colmap_images(images_txt):
    """Parse COLMAP images.txt to dict of {img_name: 4x4_pose}"""
    poses = {}
    with open(images_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 10:
                img_name = parts[9]
                qvec = np.array([float(x) for x in parts[1:5]])
                tvec = np.array([float(x) for x in parts[5:8]])

                R = qvec2rotmat(qvec)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = tvec
                poses[img_name] = T
    return poses


def parse_camera_intrinsics(cameras_txt):
    """Extract camera intrinsics as 3x3 matrix"""
    with open(cameras_txt) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                params = [float(x) for x in line.split()[4:]]
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                return np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
    return None


# ============ 主处理流程 ============
def process_scene(scene_name):
    src_scene = INPUT_DIR / scene_name
    dst_scene = OUTPUT_DIR / scene_name

    if not src_scene.exists():
        print(f"⚠️  Skip {scene_name}: source not found")
        return

    print(f"📦 Processing {scene_name}...")
    dst_scene.mkdir(parents=True, exist_ok=True)

    # 1. 重组目录结构
    src_calib = src_scene / "dslr_calibration_undistorted"
    dst_poses = dst_scene / "poses"

    if src_calib.exists():
        if dst_poses.exists():
            shutil.rmtree(dst_poses)
        shutil.copytree(src_calib, dst_poses)

    # 2. 移动图片
    src_imgs = src_scene / "images" / "dslr_images_undistorted"
    dst_imgs = dst_scene / "images"
    dst_imgs.mkdir(exist_ok=True)

    if src_imgs.exists():
        for img in src_imgs.glob("*.JPG"):
            shutil.copy2(img, dst_imgs / img.name)

    # 3. 复制扫描数据
    src_scan = src_scene / "dslr_scan_eval"
    dst_scan = dst_scene / "dslr_scan_eval"
    if src_scan.exists():
        if dst_scan.exists():
            shutil.rmtree(dst_scan)
        shutil.copytree(src_scan, dst_scan)

    # 4. 转换COLMAP poses为单独的4x4矩阵文件
    images_txt = dst_poses / "images.txt"
    if images_txt.exists():
        poses = parse_colmap_images(images_txt)
        valid_imgs = {img.stem for img in dst_imgs.glob("*.JPG")}

        for img_name, pose in poses.items():
            stem = Path(img_name).stem
            if stem in valid_imgs:
                pose_file = dst_poses / f"{stem}.txt"
                np.savetxt(pose_file, pose, fmt='%.8f')

    # 5. 创建intrins目录并保存3x3内参矩阵
    cameras_txt = dst_poses / "cameras.txt"
    intrins_dir = dst_scene / "intrins"
    intrins_dir.mkdir(exist_ok=True)

    if cameras_txt.exists():
        K = parse_camera_intrinsics(cameras_txt)
        if K is not None:
            for img in dst_imgs.glob("*.JPG"):
                intrin_file = intrins_dir / f"{img.stem}.txt"
                np.savetxt(intrin_file, K, fmt='%.6f')

    img_count = len(list(dst_imgs.glob("*.JPG")))
    pose_count = len(list(dst_poses.glob("DSC_*.txt")))
    intrin_count = len(list(intrins_dir.glob("*.txt")))

    print(f"   ✓ {img_count} images, {pose_count} poses, {intrin_count} intrinsics")


# ============ 执行 ============
if __name__ == "__main__":
    print(f"Source: {INPUT_DIR}")
    print(f"Target: {OUTPUT_DIR}\n")

    for scene in SCENES:
        process_scene(scene)

    print("\n✅ Done!")