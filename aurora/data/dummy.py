from torch.utils.data import Dataset
import torch
from datetime import datetime

from aurora import Batch, Metadata


class DummyDataset(Dataset):

    def __init__(self,
                 num_samples: int,
                 surf_vars: tuple[str, ...] = ("2t", "10u", "10v", "msl", "tp"),
                 atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q"),
                 static_vars: tuple[str, ...] = ("lsm", "z", "slt"),
                 atmos_levels: tuple[int, ...] = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000),
                 dim_lat: int = 16,
                 dim_lon: int = 32,
                 rollout_steps: int = 1,
                 no_xarray: bool = False,  # For compatibility with CERRADataset
                 boundary_size: tuple[int, int, int, int] = (0, 0, 0, 0),  # For compatibility with CERRADataset
                 use_evaluation_mode: bool = False,
                 ):
        self.num_samples = num_samples
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.static_vars = static_vars
        self.atmos_levels = atmos_levels
        self.dim_lat = dim_lat
        self.dim_lon = dim_lon
        self.rollout_steps = rollout_steps
        self.use_evaluation_mode = use_evaluation_mode

        # Create stats dict with 0 mean and 1 std for all variables
        self.stats = {}

        for var in self.surf_vars:
            self.stats[f"{var}"] = (0, 1)

        for var in self.atmos_vars:
            for level in self.atmos_levels:
                self.stats[f"{var}_{level}"] = (0, 1)

        for var in self.static_vars:
            self.stats[f"{var}"] = (0, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate data with 3 dimensions for surf_vars and atmos_vars
        data = Batch(
            surf_vars={
                k: torch.randn(1, 2, self.dim_lat, self.dim_lon)  # (b, t, h, w)
                for k in self.surf_vars
            },
            static_vars={
                k: torch.randn(self.dim_lat, self.dim_lon)  # (h, w)
                for k in self.static_vars
            },
            atmos_vars={
                k: torch.randn(1, 2, 13, self.dim_lat, self.dim_lon)  # (b, t, c, h, w)
                for k in self.atmos_vars
            },
            metadata=Metadata(
                lat=torch.linspace(90, -90, self.dim_lat),
                lon=torch.linspace(0, 360, self.dim_lon + 1)[:-1],
                time=(datetime(2020, 6, 1, 12, 0), ), # Single value for time
                atmos_levels=self.atmos_levels,
            )
        )

        output = {"input": data}

        if self.use_evaluation_mode:
            return output

        for i in range(0, self.rollout_steps):
            target = Batch(
                surf_vars={
                    k: torch.randn(1, 1, self.dim_lat, self.dim_lon)  # (b, t, h, w)
                    for k in self.surf_vars
                },
                static_vars={
                    k: torch.randn(self.dim_lat, self.dim_lon)  # (h, w)
                    for k in self.static_vars
                },
                atmos_vars={
                    k: torch.randn(1, 1, 13, self.dim_lat, self.dim_lon)  # (b, t, c, h, w)
                    for k in self.atmos_vars
                },
                metadata=Metadata(
                    lat=torch.linspace(90, -90, self.dim_lat),
                    lon=torch.linspace(0, 360, self.dim_lon + 1)[:-1],
                    time=(datetime(2020, 6, 1, 12, 0), ), # Single value for time
                    atmos_levels=self.atmos_levels,
                )
            )
            output[f"target_{i}"] = target

        return output
