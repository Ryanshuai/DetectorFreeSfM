import os
import os.path as osp
from pathlib import Path

from . import generate_empty
from third_party.Hierarchical_Localization.hloc import reconstruction, triangulation


def coarse_SfM_runner(
    img_list,
    img_pairs,
    coarse_dir,
    image_dir,
    match_folder,
    colmap_configs=None,
    triangulation_mode=False,
    prior_intrin_path=None,
    prior_pose_path=None,
    prior_model_path=None,
    verbose=True
):
    feature_out = osp.join(match_folder, 'keypoints.h5')
    match_out = osp.join(match_folder, 'matches.h5')

    os.makedirs(coarse_dir, exist_ok=True)
    pair_path = osp.join(coarse_dir, 'pairs.txt')
    with open(pair_path, "w") as f:
        for img_pair in img_pairs:
            img0_path, img1_path = img_pair.split(" ")
            img0_name = osp.basename(img0_path)
            img1_name = osp.basename(img1_path)

            # Load matches
            f.write(img0_name + " " + img1_name + "\n")

    if not triangulation_mode:
        reconstruction.main(Path(coarse_dir), Path(image_dir), Path(pair_path), Path(feature_out),
                            Path(match_out), Path(prior_intrin_path) if prior_intrin_path is not None else None,
                            verbose=verbose, colmap_configs=colmap_configs)
    else:
        # Prepare reference SfM model
        reference_sfm_model = osp.join(coarse_dir, 'sfm_empty')
        generate_empty.generate_model(
            img_list,
            reference_sfm_model,
            prior_colmap_model_path=prior_model_path,
            prior_pose_path=prior_pose_path,
            prior_intrin_path=prior_intrin_path,
            single_camera=colmap_configs["ImageReader_single_camera"],
        )

        triangulation.main(Path(coarse_dir), Path(reference_sfm_model), Path(image_dir), Path(pair_path),
                           Path(feature_out), Path(match_out), colmap_configs=colmap_configs, verbose=verbose)
