import torch
from torch import nn
import torch.nn.functional as F


class RoIAlign(nn.Module):
    def __init__(self, crop_height, crop_width, extrapolation_value=0, transform_fpcoor=True):
        super(RoIAlign, self).__init__()
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.transform_fpcoor = transform_fpcoor
        assert extrapolation_value == 0, \
            "extrapolation_value must be 0 (PyTorch grid_sample limitation)"

    def forward(self, featuremap, boxes, box_ind):
        """
        Pure PyTorch RoIAlign implementation, no CUDA compilation required.

        Args:
            featuremap: (N, C, H, W) input feature maps
            boxes: (M, 4) [x1, y1, x2, y2] unnormalized coordinates
            box_ind: (M,) batch indices for each box

        Returns:
            (M, C, crop_height, crop_width) cropped features
        """
        x1, y1, x2, y2 = boxes.split(1, dim=1)
        H, W = featuremap.shape[2:4]

        # Normalize coordinates to [0, 1]
        if self.transform_fpcoor:
            spacing_w = (x2 - x1) / self.crop_width
            spacing_h = (y2 - y1) / self.crop_height
            nx0 = (x1 + spacing_w / 2 - 0.5) / (W - 1)
            ny0 = (y1 + spacing_h / 2 - 0.5) / (H - 1)
            nw = spacing_w * (self.crop_width - 1) / (W - 1)
            nh = spacing_h * (self.crop_height - 1) / (H - 1)
            boxes_norm = torch.cat([ny0, nx0, ny0 + nh, nx0 + nw], 1)
        else:
            boxes_norm = torch.cat([
                y1 / (H - 1), x1 / (W - 1),
                y2 / (H - 1), x2 / (W - 1)
            ], 1)

        # Generate sampling grid
        y_grid = torch.linspace(0, 1, self.crop_height, device=featuremap.device)
        x_grid = torch.linspace(0, 1, self.crop_width, device=featuremap.device)
        grid_y, grid_x = torch.meshgrid(y_grid, x_grid, indexing='ij')

        # Crop features using bilinear interpolation
        crops = []
        for i in range(boxes.shape[0]):
            y1, x1, y2, x2 = boxes_norm[i]
            b_idx = box_ind[i].long()

            # Map [0,1] grid to box region, then to [-1,1] for grid_sample
            sample_y = y1 + (y2 - y1) * grid_y
            sample_x = x1 + (x2 - x1) * grid_x
            grid = torch.stack([
                2 * sample_x - 1,
                2 * sample_y - 1
            ], dim=-1).unsqueeze(0)

            crop = F.grid_sample(
                featuremap[b_idx:b_idx + 1], grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
            crops.append(crop)

        return torch.cat(crops, dim=0)