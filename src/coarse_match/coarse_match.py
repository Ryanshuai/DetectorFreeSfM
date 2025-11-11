import os.path as osp

from src.utils.data_io import save_h5
from .kornia_loftr import LoFTRMatcher
from .match_to_track import matches_to_indexed_tracks

cfgs = {
    "data": {
        "img_resize": 1600,
        "df": 8,
        "pad_to": None,
        "img_type": "grayscale",  # ['grayscale', 'rgb']
        "img_preload": False,
    },
    "matcher": {
        "model": {
            "matcher": 'loftr_official',
            "type": "coarse_only",  # ['coarse_only', 'coarse_fine]
            "match_thr": 0.2,
            "matchformer":
                {
                    "cfg_path_coarse_only": "third_party/MatchFormer/config/matchformer_coarse_only.py",
                    "cfg_path_coarse_fine": "third_party/MatchFormer/config/matchformer_coarse_fine.py",
                    "weight_path": "weight/outdoor-large-LA.ckpt",
                },
            "aspanformer": {
                "cfg_path_coarse_only": "third_party/aspantransformer/configs/aspan/outdoor/aspan_test_coarse_only.py",
                "cfg_path_coarse_fine": "third_party/aspantransformer/configs/aspan/outdoor/aspan_test.py",
                "weight_path": "weight/aspanformer_weights/outdoor.ckpt",
            },
            "loftr_official":
                {
                    "cfg_path_coarse_only": "third_party/LoFTR/configs/loftr/outdoor/loftr_ds_coarse_only.py",
                    "cfg_path_coarse_fine": "third_party/LoFTR/configs/loftr/outdoor/loftr_ds.py",
                    "weight_path": "weight/outdoor_ds.ckpt",
                },
            "seed": 666
        },
        "round_matches_ratio": 4,
        "pair_name_split": " ",
    },
    "coarse_match_debug": True,
    "ray": {
        "slurm": False,
        "n_workers": 8,  # 16
        "n_cpus_per_worker": 2,
        "n_gpus_per_worker": 0.5,
        "local_mode": False,
    },
}


def detector_free_coarse_matching(
    image_dir,
    image_lists,
    pair_list,
    match_result_folder,
    verbose=True
):
    kornia_loftr = LoFTRMatcher()
    kornia_loftr.prepare_data(cfgs["data"], image_dir, pair_list)
    matches = kornia_loftr.match_all_pairs()

    keypoints, scores, match_indices = matches_to_indexed_tracks(matches, image_lists)

    cache_dir = osp.join(match_result_folder, "raw_matches.h5")
    save_h5(matches, cache_dir, verbose=verbose)
    feature_out = osp.join(match_result_folder, "keypoints.h5")
    save_h5(keypoints, feature_out)
    match_out = osp.join(match_result_folder, "matches.h5")
    save_h5(match_indices, match_out)

    return keypoints, match_indices
