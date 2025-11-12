import os
import os.path as osp
from pathlib import Path
import shutil

from loguru import logger
import natsort

from src.utils.data_io import save_h5
from .coarse_match.kornia_loftr import LoFTRMatcher
from .coarse_match.match_to_track import matches_to_indexed_tracks
from src.evaluator import Evaluator
from src.construct_pairs import construct_img_pairs
from third_party.Hierarchical_Localization.hloc import reconstruction
from .post_optimization.post_optimization_class import SfMRefiner, sfm_refiner_cfg
from .post_optimization.utils.write_fixed_images import fix_farest_images
from src.sfm_runner.reregistration import run_image_reregistration


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

    matching_folder = osp.join(work_dir, method_name, "matching")
    matching_feature_pth = osp.join(matching_folder, 'keypoints.h5')
    matching_match_pth = osp.join(matching_folder, 'matches.h5')
    matching_raw_match_pth = osp.join(matching_folder, 'raw_matches.h5')

    coarse_dir = osp.join(work_dir, method_name, "coarse")
    coarse_best_model_dir = osp.join(coarse_dir, "0")
    coarse_pair_path = osp.join(coarse_dir, 'pairs.txt')
    coarse_database_path = osp.join(coarse_dir, "database.db")

    refine_1_folder = osp.join(work_dir, method_name, "refined_1")
    refine_1_fix_images_pth = osp.join(refine_1_folder, "fixed_images.txt")
    refine_1_database_path = osp.join(refine_1_folder, "database.db")
    refine_1_track_refine_folder = osp.join(refine_1_folder, "track_refine")
    refine_1_geometry_refine_folder = osp.join(refine_1_folder, "geometry_refine")
    refine_2_folder = osp.join(work_dir, method_name, "refined_2")
    refine_2_fix_images_pth = osp.join(refine_2_folder, "fixed_images.txt")
    refine_2_database_path = osp.join(refine_2_folder, "database.db")
    refine_2_track_refine_folder = osp.join(refine_2_folder, "track_refine")
    refine_2_geometry_refine_folder = osp.join(refine_2_folder, "geometry_refine")

    os.makedirs(osp.join(work_dir, method_name), exist_ok=True)
    os.makedirs(coarse_dir, exist_ok=True)
    with open(coarse_pair_path, "w") as f:
        f.write("\n".join(img_pairs))

    coarse_matching_data_cfgs = {
        "img_resize": 1600,
        "df": 8,
        "pad_to": None,
        "img_type": "grayscale",  # ['grayscale', 'rgb']
        "img_preload": False,
    }

    print("""
    # ==================================================================================================================
    # Matching
    # ==================================================================================================================
    """)

    kornia_loftr = LoFTRMatcher()
    kornia_loftr.prepare_data(coarse_matching_data_cfgs, img_dir, img_pairs)
    matches = kornia_loftr.match_all_pairs()

    keypoints, scores, match_indices = matches_to_indexed_tracks(matches, img_names)

    save_h5(matches, matching_raw_match_pth)
    save_h5(keypoints, matching_feature_pth)
    save_h5(match_indices, matching_match_pth)

    print("""
    # ==================================================================================================================
    # Coarse SfM Reconstruction
    # ==================================================================================================================
    """)

    reconstruction.main(sfm_dir=Path(coarse_dir),
                        image_dir=Path(img_dir),
                        pairs=Path(coarse_pair_path),
                        features=Path(matching_feature_pth),
                        matches=Path(matching_match_pth),
                        prior_intrin=None, verbose=verbose, colmap_configs=colmap_configs)

    print("""
    # ==================================================================================================================
    # Refine 1
    # ==================================================================================================================
    """)

    refiner = SfMRefiner(
        sfm_refiner_cfg,
        matcher_model_path=args.NEUSFM_fine_match_model_path,
        matcher_cfg_path=args.NEUSFM_fine_match_cfg_path,
        chunk_size=args.NEUSFM_refinement_chunk_size,
        colmap_configs=colmap_configs,
    )
    refine_3D_pts_only = False
    fix_farest_images(coarse_best_model_dir, refine_1_fix_images_pth)

    shutil.copy2(coarse_database_path, refine_1_database_path)

    refiner.refine_iteration(
        img_dir,
        img_names,
        img_pairs,
        fixed_image_txt=refine_1_fix_images_pth,
        input_model_dir=coarse_best_model_dir,
        output_track_model_dir=refine_1_track_refine_folder,
        output_geometry_model_dir=refine_1_geometry_refine_folder,
        database_path=refine_1_database_path,
        only_basename_in_colmap=True,
        rewindow_size_factor=0,  # TODO i*2
        filter_threshold=refiner.cfgs['incremental_refiner_filter_thresholds'][0],
        refine_3D_pts_only=refine_3D_pts_only,
        colmap_configs=refiner.colmap_configs,
        verbose=verbose
    )

    print("""
    # ==================================================================================================================
    # Refine 2
    # ==================================================================================================================
    """)

    if not refine_3D_pts_only:
        run_image_reregistration(input_model_dir=refine_1_geometry_refine_folder,
                                 output_model_dir=refine_1_geometry_refine_folder,
                                 database_path=refine_1_database_path,
                                 colmap_configs=colmap_configs)

    os.makedirs(refine_2_folder, exist_ok=True)
    shutil.copy2(refine_1_fix_images_pth, refine_2_fix_images_pth)
    shutil.copy2(refine_1_database_path, refine_2_database_path)

    # fix_farest_images(refine_1_geometry_refine_folder, refine_2_fix_images_pth)

    refiner.refine_iteration(
        img_dir,
        img_names,
        img_pairs,
        fixed_image_txt=refine_2_fix_images_pth,
        input_model_dir=refine_1_geometry_refine_folder,
        output_track_model_dir=refine_2_track_refine_folder,
        output_geometry_model_dir=refine_2_geometry_refine_folder,
        database_path=refine_2_database_path,
        only_basename_in_colmap=True,
        rewindow_size_factor=2,  # TODO i*2
        filter_threshold=refiner.cfgs['incremental_refiner_filter_thresholds'][1],
        refine_3D_pts_only=refine_3D_pts_only,
        colmap_configs=refiner.colmap_configs,
        verbose=verbose
    )

    print("""
    # ==================================================================================================================
    # Evaluation
    # ==================================================================================================================
    """)

    model_dirs = [
        coarse_best_model_dir,
        refine_1_geometry_refine_folder,
        refine_2_geometry_refine_folder,
    ]

    def evaluate_models(evaluator, model_dirs, verbose=True):
        for model_dir in model_dirs:
            logger.info(f"Evaluating: {model_dir}") if verbose else None
            error_dict, metrics_dict = evaluator.eval_metric(model_dir)
        return evaluator.prepare_output_from_buffer()

    evaluator = (
        Evaluator(img_names, gt_pose_dir, triangulate_mode=False, verbose=verbose)
        if not args.close_eval and gt_pose_dir is not None
        else None
    )

    metrics_dict = evaluate_models(evaluator, model_dirs, verbose)
    return metrics_dict
