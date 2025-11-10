from loguru import logger
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import numpy as np
from tqdm import tqdm

from src.utils.misc import lower_config
from .kornia_loftr import LoFTRMatcher
from .utils.detector_wrapper import DetectorWrapper
from ..dataset.coarse_matching_dataset import CoarseMatchingDataset


def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))


def build_model(args):
    pl.seed_everything(args['seed'])
    logger.info(f"Use {args['matcher']} as coarse matcher")
    model_type = args['type']
    match_thr = args['match_thr']
    if args['matcher'] == "loftr_official":
        from third_party.LoFTR.src.loftr.loftr import LoFTR
        from third_party.LoFTR.src.config.default import get_cfg_defaults as get_cfg_defaults_loftr

        matcher_args = args['loftr_official']
        cfg = get_cfg_defaults_loftr()
        cfg.merge_from_file(matcher_args[f'cfg_path_{model_type}'])
        match_cfg = lower_config(cfg)
        match_cfg['loftr']['match_coarse']['thr'] = match_thr
        match_cfg['loftr']['coarse']['temp_bug_fix'] = False
        matcher = LoFTR(config=match_cfg['loftr'])
        # load checkpoints
        state_dict = torch.load(matcher_args['weight_path'], map_location="cpu")["state_dict"]
        matcher.load_state_dict(state_dict, strict=True)

        detector = DetectorWrapper()
        detector.eval()
        matcher.eval()

    elif args['matcher'] == 'aspanformer':
        from third_party.aspantransformer.src.ASpanFormer.aspanformer import ASpanFormer
        from third_party.aspantransformer.src.config.default import get_cfg_defaults as get_cfg_defaults_aspanformer

        matcher_args = args['aspanformer']
        config = get_cfg_defaults_aspanformer()
        config.merge_from_file(matcher_args[f'cfg_path_{model_type}'])
        _config = lower_config(config)
        _config['aspan']['match_coarse']['thr'] = match_thr
        matcher = ASpanFormer(config=_config['aspan'], online_resize=True)
        state_dict = torch.load(matcher_args['weight_path'], map_location='cpu')['state_dict']
        matcher.load_state_dict(state_dict, strict=False)

        detector = DetectorWrapper()
        detector.eval()
        matcher.eval()

    elif args['matcher'] == 'matchformer':
        from third_party.MatchFormer.model.matchformer import Matchformer
        from third_party.MatchFormer.config.defaultmf import get_cfg_defaults as get_cfg_defaults_matchformer

        matcher_args = args['matchformer']
        config = get_cfg_defaults_matchformer()
        config.merge_from_file(matcher_args[f'cfg_path_{model_type}'])
        _config = lower_config(config)
        _config['matchformer']['match_coarse']['thr'] = match_thr
        matcher = Matchformer(config=_config['matchformer'], )
        state_dict = torch.load(matcher_args['weight_path'], map_location='cpu')
        matcher.load_state_dict(state_dict, strict=True)

        detector = DetectorWrapper()
        detector.eval()
        matcher.eval()
    else:
        raise NotImplementedError

    return detector, matcher


def extract_preds(data):
    """extract predictions assuming bs==1"""
    m_bids = data["m_bids"].cpu().numpy()
    assert (np.unique(m_bids) == 0).all()
    mkpts0 = data["mkpts0_f"].cpu().numpy()
    mkpts1 = data["mkpts1_f"].cpu().numpy()
    mconfs = data["mconf"].cpu().numpy()

    return mkpts0, mkpts1, mconfs


@torch.no_grad()
def extract_matches(data, detector=None, matcher=None):
    detector(data)
    matcher(data)

    mkpts0, mkpts1, mconfs = extract_preds(data)
    return (mkpts0, mkpts1, mconfs)


@torch.no_grad()
def match_worker(subset_ids, image_lists, covis_pairs_out, cfgs, pba=None, verbose=True):
    """extract matches from part of the possible image pair permutations"""
    args = cfgs['matcher']
    detector, matcher = build_model(args['model'])

    kornia_loftr = LoFTRMatcher()
    detector.cuda()
    matcher.cuda()
    matches = {}
    # Build dataset:
    dataset = CoarseMatchingDataset(cfgs["data"], image_lists, covis_pairs_out, subset_ids)
    dataloader = DataLoader(dataset, num_workers=4, pin_memory=True)

    tqdm_disable = True
    if not verbose:
        assert pba is None
    else:
        if pba is None:
            tqdm_disable = False

    # match all permutations
    for data in tqdm(dataloader, disable=tqdm_disable):
        f_name0, f_name1 = data['pair_key'][0][0], data['pair_key'][1][0]
        data_c = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()
        }
        mkpts0, mkpts1, mconfs = extract_matches(
            data_c,
            detector=detector,
            matcher=matcher,
        )
        mkpts0_new, mkpts1_new, mconfs_new = kornia_loftr(data_c['image0'], data_c['image1'])
        mkpts0_new = mkpts0_new * data['scale0'].cpu().numpy()[:, [1, 0]]
        mkpts1_new = mkpts1_new * data['scale1'].cpu().numpy()[:, [1, 0]]

        mkpts0, mkpts1, mconfs = (mkpts0_new, mkpts1_new, mconfs_new)

        # Round mkpts to grid-level to construct feature tracks for the later SfM
        if args['model']['type'] is not 'coarse_only' and args['round_matches_ratio'] is not None:
            mkpts0 = np.round((mkpts0 / data['scale0'][:, [1, 0]]) / args['round_matches_ratio']) * args[
                'round_matches_ratio'] * data['scale0'][:, [1, 0]]
            mkpts1 = np.round((mkpts1 / data['scale1'][:, [1, 0]]) / args['round_matches_ratio']) * args[
                'round_matches_ratio'] * data['scale1'][:, [1, 0]]

        # Extract matches (kpts-pairs & scores)
        matches[args['pair_name_split'].join([f_name0, f_name1])] = np.concatenate(
            [mkpts0, mkpts1, mconfs[:, None]], -1
        )  # (N, 5)

        if pba is not None:
            pba.update.remote(1)
    return matches
