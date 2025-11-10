import os.path as osp
from typing import ChainMap
import os

import math

from src.utils.ray_utils import split_dict
from src.utils.data_io import save_h5, load_h5
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
    covis_pairs_out,
    feature_out,
    match_out,
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

    # Construct directory
    base_dir = feature_out.rsplit("/", 1)[0]
    os.makedirs(base_dir, exist_ok=True)
    cache_dir = osp.join(feature_out.rsplit("/", 1)[0], "raw_matches.h5")

    if isinstance(covis_pairs_out, list):
        pair_list = covis_pairs_out
    else:
        assert osp.exists(covis_pairs_out)
        # Load pairs: 
        with open(covis_pairs_out, 'r') as f:
            pair_list = f.read().rstrip('\n').split('\n')

    # Matcher runner
    if not cfgs["coarse_match_debug"] and osp.exists(cache_dir):
        matches = load_h5(cache_dir, transform_slash=True)
        logger.info("Caches raw matches loaded!")
    else:
        all_ids = np.arange(0, len(pair_list))

        matches = match_worker(all_ids, image_lists, covis_pairs_out, cfgs, verbose=verbose)
        logger.info("Matcher finish!")

        logger.info(f"Raw matches cach begin: {cache_dir}")
        save_h5(matches, cache_dir, verbose=verbose)

    image_keypoint_to_index, match_indices = matches_to_indexed_tracks(matches, image_lists)

    updated_matches = match_indices
    keypoints = image_keypoint_to_index

    # Post process keypoints:
    keypoints = {
        k: v for k, v in keypoints.items() if isinstance(v, dict)
    }
    logger.info("Post-processing keypoints...")
    kpts_scores = [
        transform_keypoints(sub_kpts, verbose=verbose)
        for sub_kpts in split_dict(keypoints, math.ceil(len(keypoints) / 1))
    ]
    final_keypoints = dict(ChainMap(*[k for k, _ in kpts_scores]))
    final_scores = dict(ChainMap(*[s for _, s in kpts_scores]))

    # Reformat keypoints_dict and matches_dict
    # from (abs_img_path0 abs_img_path1) -> (img_name0, img_name1)
    keypoints_renamed = {}
    for key, value in final_keypoints.items():
        keypoints_renamed[osp.basename(key)] = value

    matches_renamed = {}
    for key, value in updated_matches.items():
        name0, name1 = key.split(cfgs["matcher"]["pair_name_split"])
        new_pair_name = cfgs["matcher"]["pair_name_split"].join(
            [osp.basename(name0), osp.basename(name1)]
        )
        matches_renamed[new_pair_name] = value.T

    save_h5(keypoints_renamed, feature_out)
    save_h5(matches_renamed, match_out)

    return final_keypoints, updated_matches
