"""
Adapted from: https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/decoders.py
License: https://github.com/Sentinal4D/cellshape-cloud/blob/main/cellshape_cloud/vendor/LICENSE_TearingNet
"""

import numpy as np
import torch
from torch import nn


class FoldingNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_output_points: int,
        hidden_dim: int = 512,
        std: float = 0.3,
        shape: str = "plane",
        sphere_path: str = "",
        gaussian_path: str = "",
        num_coords: int = 3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_output_points = num_output_points
        self.shape = shape
        self.sphere_path = sphere_path
        self.num_coords = num_coords
        self.gaussian_path = gaussian_path

        # make grid
        if self.shape == "plane":
            self.grid_dim = 2
            # grid_side = np.sqrt(num_output_points).astype(int)
            # range_x = torch.linspace(-std, std, grid_side)
            # range_y = torch.linspace(-std, std, grid_side)
            # x_coor, y_coor = torch.meshgrid(range_x, range_y, indexing="ij")
            # self.grid = torch.stack([x_coor, y_coor], axis=-1).float().reshape(-1, 2)

            grid_side = int(np.ceil(np.sqrt(num_output_points)))
            range_x = torch.linspace(-std, std, grid_side)
            range_y = torch.linspace(-std, std, grid_side)
            
            x_coor, y_coor = torch.meshgrid(range_x, range_y, indexing="ij")
            grid = torch.stack([x_coor, y_coor], axis=-1).float().reshape(-1, 2)
            
            if grid.shape[0] > num_output_points:
                self.grid = grid[:num_output_points]
            elif grid.shape[0] < num_output_points:
                raise ValueError("num_output_points must be a perfect square or handled explicitly.")
            else:
                self.grid = grid
        elif self.shape == "sphere":
            self.grid_dim = 3
            self.grid = torch.tensor(np.load(self.sphere_path)).float()
        elif self.shape == "gaussian":
            self.grid_dim = 3
            self.grid = torch.tensor(np.load(self.gaussian_path)).float()

        self.hidden_dim = hidden_dim

        if input_dim != hidden_dim:
            self.project = nn.Linear(input_dim, hidden_dim, bias=False)
        else:
            self.project = nn.Identity()

        self.folding1 = nn.Sequential(
            nn.Linear(hidden_dim + self.grid_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_coords),
        )

        self.folding2 = nn.Sequential(
            nn.Linear(hidden_dim + self.num_coords, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_coords),
        )

    def forward(self, x):
        x = self.project(x) # [B, hidden_dim]

        grid = self.grid.unsqueeze(0).expand(x.shape[0], -1, -1) # [B, output_points - 23 (?), 2]
        grid = grid.type_as(x)
        x = x.unsqueeze(1)
        cw_exp = x.expand(-1, grid.shape[1], -1) # [B, 2025, hidden]

        cat1 = torch.cat((cw_exp, grid), dim=2) # [B, 2025, hidden+2]
        folding_result1 = self.folding1(cat1) # [B, 2025, 3] (2D) | [B, 2025, 4] (3D)
        cat2 = torch.cat((cw_exp, folding_result1), dim=2) # [B, 2025, 515] (2D) | [B, 2025, 516] (3D)
        folding_result2 = self.folding2(cat2)
        # [B, 2025, 3] (2D) | [B, 2025, 4] (3D)

        return folding_result2
