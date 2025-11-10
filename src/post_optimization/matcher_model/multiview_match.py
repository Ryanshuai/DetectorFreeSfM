from .multiview_match_worker import build_model, matchWorker


def multiview_matcher(cfgs, dataset_cfgs, colmap_image_dataset, rewindow_size_factor=None, model_idx=None,
                      visualize_dir=None, verbose=True):
    matcher = build_model(cfgs["model"], rewindow_size_factor, model_idx)

    fine_match_results = matchWorker(
        colmap_image_dataset,
        matcher,
        subset_track_idxs=None,
        visualize=cfgs["visualize"],
        visualize_dir=visualize_dir,
        dataset_cfgs=dataset_cfgs,
        verbose=verbose
    )
    return fine_match_results
