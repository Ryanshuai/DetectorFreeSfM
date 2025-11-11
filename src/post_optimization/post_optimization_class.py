import os
import os.path as osp

from src.sfm_runner import sfm_model_geometry_refiner_new
from src.sfm_runner import reregistration
from ..dataset.coarse_sfm_refinement_dataset import CoarseColmapDataset
from .matcher_model import *
from .utils.write_fixed_images import fix_farest_images, fix_all_images

cfgs = {
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
        img_preload=False,
        refine_iter_n_times=2,
        use_pba=True,
        colmap_configs=None,
    ):
        self.cfgs = cfgs
        self.colmap_configs = colmap_configs or {}

        self.cfgs['refine_iter_n_times'] = refine_iter_n_times
        self.cfgs['fine_matcher']['model']['cfg_path'] = matcher_cfg_path
        self.cfgs['fine_matcher']['model']['weight_path'] = matcher_model_path
        self.cfgs['coarse_colmap_data']['img_preload'] = img_preload
        self.cfgs['multiview_matcher_data']['chunk'] = chunk_size
        self.cfgs['incremental_refiner_use_pba'] = self.colmap_configs.get('use_pba', use_pba)

    def generate_fixed_image(self, refine_3D_pts_only, colmap_coarse_dir, colmap_refined_kpts_dir):
        os.makedirs(colmap_refined_kpts_dir, exist_ok=True)
        os.system(
            f"cp {osp.join(osp.dirname(colmap_coarse_dir), 'database.db')} {osp.join(colmap_refined_kpts_dir, 'database.db')}")
        if refine_3D_pts_only:
            fix_all_images(reconstructed_model_dir=colmap_coarse_dir,
                           output_path=osp.join(colmap_refined_kpts_dir, 'fixed_images.txt'))
        else:
            fix_farest_images(reconstructed_model_dir=colmap_coarse_dir,
                              output_path=osp.join(colmap_refined_kpts_dir, 'fixed_images.txt'))

    def refine_iteration(self, image_lists, covis_pairs_pth, input_model_dir, stage_output_dir,
                         only_basename_in_colmap, rewindow_size_factor, filter_threshold,
                         image_path, refine_3D_pts_only, colmap_configs, verbose):
        os.makedirs(stage_output_dir, exist_ok=True)
        colmap_image_dataset = CoarseColmapDataset(
            cfgs["coarse_colmap_data"],
            image_lists,
            covis_pairs_pth,
            input_model_dir,
            stage_output_dir,
            only_basename_in_colmap=only_basename_in_colmap,
            vis_path=None)

        assert colmap_image_dataset.state

        logger.info(f"Multi-view refinement matching begin!")
        fine_match_results = multiview_matcher(
            cfgs["fine_matcher"],
            cfgs["multiview_matcher_data"],
            colmap_image_dataset,
            rewindow_size_factor=rewindow_size_factor,
        )

        colmap_image_dataset.update_refined_kpts_to_colmap_multiview(fine_match_results)
        colmap_image_dataset.save_colmap_model(osp.join(stage_output_dir, 'refined_kpts_model'))

        success = sfm_model_geometry_refiner_new.main(stage_output_dir, stage_output_dir,
                                                      no_filter_pts=cfgs["model_refiner_no_filter_pts"],
                                                      colmap_configs=colmap_configs, image_path=image_path,
                                                      verbose=verbose, refine_3D_pts_only=refine_3D_pts_only,
                                                      filter_threshold=filter_threshold,
                                                      use_pba=cfgs["incremental_refiner_use_pba"])
        return success

    def register_images(self, colmap_refined_kpts_dir, current_model_dir, colmap_configs):
        reregistration.main(colmap_refined_kpts_dir, current_model_dir, colmap_configs=colmap_configs)

    def refine(self, image_lists, covis_pairs_pth, colmap_coarse_dir, refined_output_dir,
               only_basename_in_colmap, image_path, refine_3D_pts_only=False, verbose=True):

        self.generate_fixed_image(refine_3D_pts_only, colmap_coarse_dir, refined_output_dir + "_1")

        self.refine_iteration(
            image_lists,
            covis_pairs_pth,
            colmap_coarse_dir,
            refined_output_dir + "_1",
            only_basename_in_colmap,
            rewindow_size_factor=0,  # TODO i*2
            filter_threshold=self.cfgs['incremental_refiner_filter_thresholds'][0],
            image_path=image_path,
            refine_3D_pts_only=refine_3D_pts_only,
            colmap_configs=self.colmap_configs,
            verbose=verbose
        )

        if not refine_3D_pts_only:
            self.register_images(
                refined_output_dir + "_1",
                os.path.join(refined_output_dir + "_1", '0'),
                self.colmap_configs
            )

        os.makedirs(f"{refined_output_dir}_2", exist_ok=True)
        os.system(f"cp {refined_output_dir}_1/fixed_images.txt {refined_output_dir}_2/")
        os.system(f"cp {refined_output_dir}_1/database.db {refined_output_dir}_2/")

        self.generate_fixed_image(refine_3D_pts_only, os.path.join(refined_output_dir + "_1", '0'),
                                  refined_output_dir + "_2")

        self.refine_iteration(
            image_lists,
            covis_pairs_pth,
            os.path.join(refined_output_dir + "_1", '0'),
            refined_output_dir + "_2",
            only_basename_in_colmap,
            rewindow_size_factor=2,  # TODO i*2
            filter_threshold=self.cfgs['incremental_refiner_filter_thresholds'][1],
            image_path=image_path,
            refine_3D_pts_only=refine_3D_pts_only,
            colmap_configs=self.colmap_configs,
            verbose=verbose
        )


def post_optimization(
    image_lists,
    covis_pairs_pth,
    colmap_coarse_dir,
    refined_model_save_dir,
    match_out_pth,
    image_path="",
    chunk_size=6000,
    matcher_model_path=None,
    matcher_cfg_path=None,
    img_resize=None,
    img_preload=False,
    colmap_configs=None,
    only_basename_in_colmap=False,
    visualize_dir=None,
    vis3d_pth=None,
    refine_iter_n_times=2,
    refine_3D_pts_only=False,
    verbose=True
):
    refiner = SfMRefiner(
        cfgs,
        matcher_model_path=matcher_model_path,
        matcher_cfg_path=matcher_cfg_path,
        chunk_size=chunk_size,
        img_preload=img_preload,
        refine_iter_n_times=refine_iter_n_times,
        colmap_configs=colmap_configs,
    )
    refiner.refine(
        image_lists,
        covis_pairs_pth,
        colmap_coarse_dir,
        refined_model_save_dir,
        only_basename_in_colmap,
        image_path=image_path,
        refine_3D_pts_only=refine_3D_pts_only,
        verbose=verbose,
    )
