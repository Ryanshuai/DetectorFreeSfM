import os.path as osp

from torch.utils.data import Dataset

from .utils import read_grayscale, read_rgb


class CoarseMatchingDataset(Dataset):
    def __init__(
        self,
        args,
        image_dir,
        image_pairs,
        subset_ids=None,
    ):
        super().__init__()
        self.pair_list = image_pairs
        self.subset_ids = subset_ids or list(range(len(image_pairs)))
        self.image_dir = image_dir

        self.img_resize = args['img_resize']
        self.df = args['df']
        self.pad_to = args['pad_to']
        self.img_dict = {}

        self.img_read_func = read_grayscale if args['img_type'] == 'grayscale' else read_rgb

    def __len__(self):
        return len(self.subset_ids)

    def __getitem__(self, idx):
        return self._get_single_item(idx)

    def _get_single_item(self, idx):
        pair_idx = self.subset_ids[idx]

        img_name0, img_name1 = self.pair_list[pair_idx].split(' ')
        img_path0 = osp.join(self.image_dir, img_name0)
        img_path1 = osp.join(self.image_dir, img_name1)
        img_scale0 = self.img_read_func(
            img_path0,
            (self.img_resize,) if self.img_resize is not None else None,
            df=self.df,
            pad_to=self.pad_to,
            ret_scales=True,
        )
        img_scale1 = self.img_read_func(
            img_path1,
            (self.img_resize,) if self.img_resize is not None else None,
            pad_to=self.pad_to,
            df=self.df,
            ret_scales=True,
        )

        img0, scale0, original_hw0 = img_scale0
        img1, scale1, original_hw1 = img_scale1

        data = {
            "image0": img0,
            "image1": img1,
            "scale0": scale0,  # 1*2
            "scale1": scale1,
            "f_name0": osp.basename(img_path0).rsplit('.', 1)[0],
            "f_name1": osp.basename(img_path1).rsplit('.', 1)[0],
            "frameID": pair_idx,
            "pair_key": (img_path0, img_path1),
        }

        return data
