import os
import os.path as osp
from pathlib import Path

from loguru import logger
import natsort

from src.utils.data_io import save_h5
from .coarse_match.kornia_loftr import LoFTRMatcher
from .coarse_match.match_to_track import matches_to_indexed_tracks
from src.evaluator import Evaluator
from src.construct_pairs import construct_img_pairs
from .post_optimization.post_optimization_class import post_optimization
from third_party.Hierarchical_Localization.hloc import reconstruction


def DetectorFreeSfM(
    args,
    work_dir,
    gt_pose_dir=None,
    prior_intrin_dir=None,
    prior_pose_dir=None,
    prior_colmap_dir=None,
    colmap_configs=None,
    verbose=True
):
    # Prepare data structure
    img_dir = osp.join(work_dir, "images")
    assert osp.exists(img_dir)
    img_names = natsort.natsorted(os.listdir(img_dir))

    img_pairs = construct_img_pairs(img_names, args, strategy=args.img_pair_strategy, verbose=verbose)

    method_name = "my_res_exp000"
    os.makedirs(osp.join(work_dir, method_name), exist_ok=True)

    matching_folder = osp.join(work_dir, method_name, "matching")
    matching_feature_pth = osp.join(matching_folder, 'keypoints.h5')
    matching_match_pth = osp.join(matching_folder, 'matches.h5')
    matching_raw_match_pth = osp.join(matching_folder, 'raw_matches.h5')

    coarse_dir = osp.join(work_dir, method_name, "coarse")
    os.makedirs(coarse_dir, exist_ok=True)
    coarse_pair_path = osp.join(coarse_dir, 'pairs.txt')
    with open(coarse_pair_path, "w") as f:
        f.write("\n".join(img_pairs))

    coarse_matching_data_cfgs = {
        "img_resize": 1600,
        "df": 8,
        "pad_to": None,
        "img_type": "grayscale",  # ['grayscale', 'rgb']
        "img_preload": False,
    }

    kornia_loftr = LoFTRMatcher()
    kornia_loftr.prepare_data(coarse_matching_data_cfgs, img_dir, img_pairs)
    matches = kornia_loftr.match_all_pairs()

    keypoints, scores, match_indices = matches_to_indexed_tracks(matches, img_names)

    save_h5(matches, matching_raw_match_pth)
    save_h5(keypoints, matching_feature_pth)
    save_h5(match_indices, matching_match_pth)

    reconstruction.main(sfm_dir=Path(coarse_dir),
                        image_dir=Path(img_dir),
                        pairs=Path(coarse_pair_path),
                        features=Path(matching_feature_pth),
                        matches=Path(matching_match_pth),
                        prior_intrin=None, verbose=verbose, colmap_configs=colmap_configs)

    best_model_id = '0'

    post_optimization(
        img_names,
        img_pairs,
        colmap_coarse_dir=osp.join(work_dir, method_name, "coarse", best_model_id),
        refined_model_save_dir=osp.join(work_dir, method_name, "refined"),
        chunk_size=args.NEUSFM_refinement_chunk_size,
        matcher_model_path=args.NEUSFM_fine_match_model_path,
        matcher_cfg_path=args.NEUSFM_fine_match_cfg_path,
        only_basename_in_colmap=True,
        colmap_configs=colmap_configs,
        refine_3D_pts_only=False,
        verbose=verbose,
        image_path=img_dir,
    )

    # ==================================================================================================================
    # Evaluation
    # ==================================================================================================================

    evaluator = (
        Evaluator(img_names, gt_pose_dir, triangulate_mode=False, verbose=verbose)
        if not args.close_eval and gt_pose_dir is not None
        else None
    )

    error_dict, metrics_dict = evaluator.eval_metric(
        osp.join(work_dir, method_name, "coarse", best_model_id, best_model_id))

    temp_refined_dirs = [osp.join(osp.join(work_dir, method_name), f"refined_{id}", "0") for id in range(1, 3)]

    for temp_dir in temp_refined_dirs:
        logger.info(f"Metric of: {temp_dir}")
        error_dict, metrics_dict = evaluator.eval_metric(
            osp.join(osp.dirname(osp.join(work_dir, method_name, "refined")), temp_dir)
        )

    logger.info(f"Metric of: Final") if verbose else None
    error_dict, metrics_dict = evaluator.eval_metric(osp.join(work_dir, method_name, "refined") + "_2/0")

    metrics_dict = evaluator.prepare_output_from_buffer()
    return metrics_dict
