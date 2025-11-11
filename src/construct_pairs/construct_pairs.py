from loguru import logger

from .pairs_exhaustive import exhaustive_all_pairs
from .pairs_from_img_index import pairs_from_index


def construct_img_pairs(img_list, args, strategy='exhaustive', verbose=True):
    logger.info(f'Using {strategy} matching build pairs') if verbose else None
    if strategy == 'exhaustive':
        img_pairs = exhaustive_all_pairs(img_list)
    elif strategy == 'pair_from_index':
        img_pairs = pairs_from_index(img_list, args.INDEX_num_of_pair)
    else:
        raise NotImplementedError

    return img_pairs
