import os
import os.path as osp

from loguru import logger
import natsort

from src.evaluator import Evaluator
from src.construct_pairs import construct_img_pairs
from .coarse_match.coarse_match import detector_free_coarse_matching
from .sfm_runner.coarse_sfm_runner_new import coarse_SfM_runner
from .post_optimization.post_optimization_class import post_optimization


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
    image_dir = osp.join(work_dir, "images")
    assert osp.exists(image_dir)
    img_names = natsort.natsorted(os.listdir(image_dir))

    img_pairs = construct_img_pairs(img_names, args, strategy=args.img_pair_strategy, verbose=verbose)

    # Parse configs
    triangulation_mode = args.triangulation_mode

    method_name = "my_res_exp000"
    os.makedirs(osp.join(work_dir, method_name), exist_ok=True)

    detector_free_coarse_matching(
        image_dir,
        img_names,
        img_pairs,
        match_result_folder=osp.join(work_dir, method_name, "matching"),
        verbose=verbose,
    )

    coarse_SfM_runner(
        img_names,
        img_pairs,
        coarse_dir=osp.join(work_dir, method_name, "coarse"),
        image_dir=image_dir,
        match_folder=osp.join(work_dir, method_name, "matching"),
        colmap_configs=colmap_configs,
        triangulation_mode=triangulation_mode,
        prior_intrin_path=prior_intrin_dir,
        prior_pose_path=prior_pose_dir if triangulation_mode else None,
        prior_model_path=prior_colmap_dir if triangulation_mode else None,
        verbose=verbose,
    )

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
        refine_3D_pts_only=triangulation_mode and not args.tri_refine_pose_and_points,
        verbose=verbose,
        image_path=image_dir,
    )

    evaluator = (
        Evaluator(img_names, gt_pose_dir, triangulate_mode=args.triangulation_mode, verbose=verbose)
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
