import logging
import subprocess
import os.path as osp


def run_image_reregistration(
    input_model_dir, output_model_dir, database_path, colmap_path="colmap", colmap_configs=None, verbose=True
):
    cmd = [
        str(colmap_path),
        "image_registrator",
        "--database_path",
        str(database_path),
        "--input_path",
        str(input_model_dir),
        "--output_path",
        str(output_model_dir),
    ]

    if colmap_configs is not None and colmap_configs["no_refine_intrinsics"] is True:
        cmd += [
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
        ]

    if 'reregistration' in colmap_configs:
        # Set to lower threshold to registrate more images
        cmd += [
            "--Mapper.abs_pose_max_error",
            str(colmap_configs['reregistration']['abs_pose_max_error']),
            "--Mapper.abs_pose_min_num_inliers",
            str(colmap_configs['reregistration']['abs_pose_min_num_inliers']),
            "--Mapper.abs_pose_min_inlier_ratio",
            str(colmap_configs['reregistration']['abs_pose_min_inlier_ratio']),
            "--Mapper.filter_max_reproj_error",
            str(colmap_configs['reregistration']['filter_max_reproj_error'])
        ]

    if verbose:
        logging.info(' '.join(cmd))
        ret = subprocess.call(cmd)
    else:
        ret_all = subprocess.run(cmd, capture_output=True)
        with open(osp.join(output_model_dir, 'reregistration_output.txt'), 'w') as f:
            f.write(ret_all.stdout.decode())
        ret = ret_all.returncode

    if ret != 0:
        logging.warning("Problem with image registration, existing.")
        exit(ret)
