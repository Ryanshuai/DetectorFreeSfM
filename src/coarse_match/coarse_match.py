import os.path as osp
import os

from src.utils.data_io import save_h5
from .coarse_match_worker import *
from .match_to_track import matches_to_indexed_tracks

cfgs = {
    "data": {
        "img_resize": 1200,
        "df": 8,
        "pad_to": None,
        "img_type": "grayscale",  # ['grayscale', 'rgb']
        "img_preload": True,
    },
    "matcher": {
        "model": {
            "matcher": 'loftr',
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
    image_lists,
    pair_list,
    output_folder,
    img_resize=None,
    img_preload=False,
    matcher='loftr',
    match_type='coarse_only',  # ['coarse_only', 'coarse_fine']
    match_thr=0.2,
    match_round_ratio=None,
    verbose=True
):
    # Cfg overwrite:
    cfgs['matcher']['model']['type'] = match_type
    cfgs['matcher']['model']['match_thr'] = match_thr
    cfgs['matcher']['model']['matcher'] = matcher
    cfgs['matcher']['round_matches_ratio'] = match_round_ratio
    cfgs['data']['img_resize'] = img_resize
    cfgs['data']['img_preload'] = img_preload
    if 'loftr' in matcher:
        cfgs['data']['df'] = 8
        cfgs['data']['pad_to'] = None
    elif matcher == 'matchformer':
        cfgs['data']['df'] = 8
        cfgs['data']['pad_to'] = -1  # Two image must with same size
    elif matcher == 'aspanformer':
        cfgs['data']['df'] = None  # Will pad inner the matching module
        cfgs['data']['pad_to'] = None

    all_ids = np.arange(0, len(pair_list))
    matches = match_worker(all_ids, image_lists, pair_list, cfgs, verbose=verbose)

    keypoints, scores, match_indices = matches_to_indexed_tracks(matches, image_lists)

    # Rename: abs_path -> basename
    keypoints_renamed = {osp.basename(k): v for k, v in keypoints.items()}
    matches_renamed = {
        cfgs["matcher"]["pair_name_split"].join([osp.basename(name0), osp.basename(name1)]): v
        for k, v in match_indices.items()
        for name0, name1 in [k.split(cfgs["matcher"]["pair_name_split"])]
    }

    cache_dir = osp.join(output_folder, "raw_matches.h5")
    save_h5(matches, cache_dir, verbose=verbose)
    feature_out = osp.join(output_folder, "keypoints.h5")
    save_h5(keypoints_renamed, feature_out)
    match_out = osp.join(output_folder, "matches.h5")
    save_h5(matches_renamed, match_out)

    return keypoints, match_indices
