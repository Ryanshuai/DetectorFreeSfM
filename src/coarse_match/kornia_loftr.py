import torch
from kornia.feature import LoFTR
from torch.utils.data import DataLoader
import numpy as np

from ..dataset.coarse_matching_dataset import CoarseMatchingDataset


class LoFTRMatcher:
    def __init__(self, pretrained: str = "outdoor", device: str = "cuda"):
        self.device = torch.device(device)
        self.model = LoFTR(pretrained=None).to(self.device).eval()
        state_dict = torch.load('weight/outdoor_ds.ckpt', map_location="cpu")["state_dict"]
        self.model.load_state_dict(state_dict, strict=True)
        self.pretrained = pretrained

    def prepare_data(self, cfgs, image_lists, image_pairs):
        dataset = CoarseMatchingDataset(cfgs, image_lists, image_pairs)
        self.dataloader = DataLoader(dataset, batch_size=2, num_workers=4, pin_memory=True)

    @torch.inference_mode()
    def loftr_inference(self, img0, img1):
        img0 = img0.to(self.device, non_blocking=True)
        img1 = img1.to(self.device, non_blocking=True)

        data = {"image0": img0, "image1": img1}
        fine_result = self.model(data)

        b_ids = data['b_ids']
        batch_res = [(data['mkpts0_c'][b_ids == i].cpu().numpy(),
                      data['mkpts1_c'][b_ids == i].cpu().numpy(),
                      data['mconf'][b_ids == i].cpu().numpy())
                     for i in range(img0.shape[0])]

        return batch_res

    def quantize_keypoints(self, keypoints, ratio=1.0):
        keypoints = np.round(keypoints / ratio) * ratio
        return keypoints

    def match_all_pairs(self):
        matches = {}
        for data in self.dataloader:
            batch_results = self.loftr_inference(data['image0'], data['image1'])
            data_scale_0 = data['scale0'].cpu().numpy()
            data_scale_1 = data['scale1'].cpu().numpy()
            for i, (mkpts0, mkpts1, mconfs) in enumerate(batch_results):
                mkpts0 = self.quantize_keypoints(mkpts0, ratio=1.0)
                mkpts1 = self.quantize_keypoints(mkpts1, ratio=1.0)

                mkpts0 *= data_scale_0[i][[1, 0]]
                mkpts1 *= data_scale_1[i][[1, 0]]
                im_0_name = data['pair_key'][0][i]
                im_1_name = data['pair_key'][1][i]
                matches[f"{im_0_name} {im_1_name}"] = np.concatenate([mkpts0, mkpts1, mconfs[:, None]], -1)
        return matches
