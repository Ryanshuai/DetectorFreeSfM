import os

from src.sfm_runner.sfm_model_geometry_refiner_new import run_incremental_model_refiner
from ..dataset.coarse_sfm_refinement_dataset import CoarseColmapDataset
from .matcher_model import *

sfm_refiner_cfg = {
    "coarse_colmap_data": {
        "img_resize": 1200,
        "df": None,
        "feature_track_assignment_strategy": "midium_scale",
        "img_preload": False,
    },
    "fine_match_debug": True,
    "multiview_matcher_data": {
        "max_track_length": 16,
        "chunk": 6000
    },
    "fine_matcher": {
        "model": {
            "cfg_path": [''],
            "weight_path": [''],
            "seed": 666,
        },
        "visualize": False,
        "extract_feature_method": "fine_match_backbone",
        "ray": {
            "slurm": False,
            "n_workers": 1,
            "n_cpus_per_worker": 1,
            "n_gpus_per_worker": 1,
            "local_mode": False,
        },
    },
    "visualize": False,
    "evaluation": False,
    "refine_iter_n_times": 2,
    "model_refiner_no_filter_pts": False,
    "first_iter_resize_img_to_half": False,
    "enable_update_reproj_kpts_to_model": False,
    "enable_adaptive_downscale_window": True,  # Down scale searching window size after each iteration, e.g., 15->11->7
    "incremental_refiner_filter_thresholds": [3, 2, 1.5],
    "incremental_refiner_use_pba": False,
    # NOTE: pba does not allow share intrins or fix extrinsics, and only allow simple_radial camer model
    "enable_multiple_models": False,
}


class SfMRefiner:
    def __init__(
        self,
        cfgs,
        matcher_model_path=None,
        matcher_cfg_path=None,
        chunk_size=6000,
        use_pba=True,
        colmap_configs=None,
    ):
        self.cfgs = cfgs
        self.colmap_configs = colmap_configs or {}

        self.cfgs['fine_matcher']['model']['cfg_path'] = matcher_cfg_path
        self.cfgs['fine_matcher']['model']['weight_path'] = matcher_model_path
        self.cfgs['multiview_matcher_data']['chunk'] = chunk_size
        self.cfgs['incremental_refiner_use_pba'] = self.colmap_configs.get('use_pba', use_pba)

    def refine_iteration(self,
                         image_dir,
                         image_names,
                         image_pairs,
                         fixed_image_txt,
                         input_model_dir,
                         output_track_model_dir,
                         output_geometry_model_dir,
                         database_path,
                         only_basename_in_colmap,
                         rewindow_size_factor,
                         filter_threshold,
                         refine_3D_pts_only,
                         colmap_configs,
                         verbose):
        os.makedirs(output_track_model_dir, exist_ok=True)
        os.makedirs(output_geometry_model_dir, exist_ok=True)

        colmap_image_dataset = CoarseColmapDataset(
            sfm_refiner_cfg["coarse_colmap_data"],
            image_dir,
            image_names,
            image_pairs,
            input_model_dir,
            only_basename_in_colmap=only_basename_in_colmap,
            vis_path=None)

        assert colmap_image_dataset.state

        logger.info(f"Multi-view refinement matching begin!")
        fine_match_results = multiview_matcher(
            sfm_refiner_cfg["fine_matcher"],
            sfm_refiner_cfg["multiview_matcher_data"],
            colmap_image_dataset,
            rewindow_size_factor=rewindow_size_factor,
        )

        colmap_image_dataset.update_refined_kpts_to_colmap_multiview(fine_match_results)
        colmap_image_dataset.save_colmap_model(output_track_model_dir)

        success = run_incremental_model_refiner(output_track_model_dir,
                                                output_geometry_model_dir,
                                                database_path,
                                                image_dir,
                                                fixed_image_txt,
                                                no_filter_pts=sfm_refiner_cfg["model_refiner_no_filter_pts"],
                                                colmap_configs=colmap_configs,
                                                verbose=verbose, refine_3D_pts_only=refine_3D_pts_only,
                                                filter_threshold=filter_threshold,
                                                use_pba=sfm_refiner_cfg["incremental_refiner_use_pba"])
        return success
