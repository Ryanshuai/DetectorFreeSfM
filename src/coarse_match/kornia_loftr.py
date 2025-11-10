import torch
from kornia.feature import LoFTR


class LoFTRMatcher:
    """
    Thin wrapper around Kornia LoFTR for efficient repeated matching.
    Usage:
        matcher = LoFTRMatcher(pretrained='outdoor')
        mkpts0, mkpts1, mconfs = matcher(data_c)
    """

    def __init__(self, pretrained: str = "outdoor", device: str = "cuda"):
        self.device = torch.device(device)
        self.model = LoFTR(pretrained=None).to(self.device).eval()
        state_dict = torch.load('weight/outdoor_ds.ckpt', map_location="cpu")["state_dict"]
        self.model.load_state_dict(state_dict, strict=True)
        self.pretrained = pretrained

    @torch.no_grad()
    def __call__(self, img0, img1):
        img0 = img0.to(self.device, non_blocking=True)
        img1 = img1.to(self.device, non_blocking=True)

        data = {"image0": img0, "image1": img1}
        self.model(data)

        # Use coarse results directly
        mkpts0 = data['mkpts0_c'].cpu().numpy()
        mkpts1 = data['mkpts1_c'].cpu().numpy()
        mconfs = data['mconf'].cpu().numpy()

        return mkpts0, mkpts1, mconfs
