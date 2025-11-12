import os.path as osp

from src.utils.data_io import save_h5
from .kornia_loftr import LoFTRMatcher
from .match_to_track import matches_to_indexed_tracks

coarse_matching_data_cfgs = {
    "data": {
        "img_resize": 1600,
        "df": 8,
        "pad_to": None,
        "img_type": "grayscale",  # ['grayscale', 'rgb']
        "img_preload": False,
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
    kornia_loftr.prepare_data(coarse_matching_data_cfgs, image_dir, pair_list)
    matches = kornia_loftr.match_all_pairs()

    keypoints, scores, match_indices = matches_to_indexed_tracks(matches, image_lists)

    cache_dir = osp.join(match_result_folder, "raw_matches.h5")
    save_h5(matches, cache_dir, verbose=verbose)
    feature_out = osp.join(match_result_folder, "keypoints.h5")
    save_h5(keypoints, feature_out)
    match_out = osp.join(match_result_folder, "matches.h5")
    save_h5(match_indices, match_out)

    return keypoints, match_indices
