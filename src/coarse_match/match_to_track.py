from typing import Dict, Tuple, List
from collections import defaultdict

import numpy as np


def matches_to_indexed_tracks(
    matches: Dict[str, np.ndarray],
    image_names: List[str],
    pair_split: str = " "
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Convert pairwise matches to indexed keypoint correspondences.

    Args:
        matches: {img0-img1: [[x0,y0,x1,y1,conf], ...], shape (N,5)}
        image_names: list of image paths
        pair_split: separator for image pair names

    Returns:
        keypoints: {img_path: {(x,y): (idx, score), ...}}
        match_indices: {img0-img1: [[idx0,idx1], ...], shape (N,2)}
    """
    # Step 1: Extract keypoints per image from all matches
    keypoints_raw = extract_keypoints_from_matches(matches, image_names, pair_split)

    # Step 2: Deduplicate and assign unique IDs
    image_keypoint_to_index = build_keypoint_to_index(keypoints_raw)

    # Step 3: Convert to arrays
    keypoints, scores = keypoint_dict_to_arrays(image_keypoint_to_index)

    # Step 4: Convert matches to indices
    match_indices = matches_to_indices(matches, image_keypoint_to_index, pair_split)

    return keypoints, scores, match_indices


def extract_keypoints_from_matches(matches, image_names, pair_split):
    image_to_points = defaultdict(list)
    for pair_key, match_data in matches.items():
        name0, name1 = pair_key.split(pair_split)
        image_to_points[name0].append(match_data[:, [0, 1, 4]])  # x0, y0, conf
        image_to_points[name1].append(match_data[:, [2, 3, 4]])  # x1, y1, conf

    image_keypoints = {}
    for name in image_names:
        image_keypoints[name] = np.vstack(image_to_points[name] or [np.empty((0, 3))])

    return image_keypoints


def build_keypoint_to_index(keypoints_raw):
    im_kpt_to_idx = {}
    for name, kpts in keypoints_raw.items():
        coord_to_score = defaultdict(float)
        for x, y, score in kpts:
            coord_to_score[(int(x), int(y))] += score

        sorted_kpts = sorted(coord_to_score.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
        im_kpt_to_idx[name] = {coord: (idx, score) for idx, (coord, score) in enumerate(sorted_kpts)}
    return im_kpt_to_idx


def keypoint_dict_to_arrays(im_kpt_to_idx):
    """Convert {(x,y): (idx, score)} to arrays [N,2] and [N]"""
    keypoints = {}
    scores = {}

    for name, kpt_dict in im_kpt_to_idx.items():
        # Sort by idx to ensure correct ordering
        sorted_items = sorted(kpt_dict.items(), key=lambda x: x[1][0])

        kpts_arr = np.array([coord for coord, _ in sorted_items], dtype=np.float32)
        scores_arr = np.array([score for _, (_, score) in sorted_items], dtype=np.float32)

        keypoints[name] = kpts_arr
        scores[name] = scores_arr

    return keypoints, scores


def matches_to_indices(matches, im_kpt_to_idx, pair_split):
    match_indices = {}
    for pair_key, match_data in matches.items():
        name0, name1 = pair_key.split(pair_split)
        kpt_to_idx_0, kpt_to_idx_1 = im_kpt_to_idx[name0], im_kpt_to_idx[name1]

        indices = []
        for x0, y0, x1, y1, _ in match_data:
            coord0, coord1 = (int(x0), int(y0)), (int(x1), int(y1))
            if coord0 in kpt_to_idx_0 and coord1 in kpt_to_idx_1:
                indices.append([kpt_to_idx_0[coord0][0], kpt_to_idx_1[coord1][0]])

        # Transpose to (2, N) format
        match_indices[pair_key] = np.array(indices, dtype=np.int32).T

    return match_indices
