import torch
from datetime import datetime
from typing import Optional, Union

from aurora import Batch, Metadata


def mse(forecast: Batch, truth: Batch, variable: str, level: Optional[int] = None, latitude_weights=None, ) -> torch.Tensor:
    """
    Calculate MSE for a specific variable.
    
    Args:
        forecast (Batch): Forecast batch containing weather variables
        truth (Batch): Truth batch containing weather variables
        variable (str): Name of the variable to calculate MSE for (e.g., '2t', 'z')
        level (Optional[int]): Pressure level for atmospheric variables (e.g., 500)
                             None for surface variables
        latitude_weights (Optional[torch tensor]): Latitude weight correction for CERRA
    
    Returns:
        torch.Tensor: MSE value for the specified variable
    """
    if variable in forecast.surf_vars:
        f_var = forecast.surf_vars[variable] if variable in forecast.surf_vars else 0
        t_var = truth.surf_vars[variable] if variable in truth.surf_vars else 0
    elif variable in forecast.atmos_vars:
        if level is None:
            raise ValueError(f"Level must be specified for atmospheric variable {variable}")
        
        try:
            f_level_idx = forecast.metadata.atmos_levels.index(level)
            t_level_idx = truth.metadata.atmos_levels.index(level)
        except ValueError:
            raise ValueError(f"Level {level} not found in atmos_levels {forecast.metadata.atmos_levels}")
        
        f_var = forecast.atmos_vars[variable]
        t_var = truth.atmos_vars[variable]
        
        f_var = f_var[:, :, f_level_idx]
        t_var = t_var[:, :, t_level_idx]
    else:
        raise ValueError(f"Variable {variable} not found in forecast batch")

    if latitude_weights is None:
        se = (f_var - t_var) ** 2
        mse = se.mean()

    else:
        weighted_se = (f_var - t_var) ** 2 * latitude_weights.to(f_var.device)
        mse = weighted_se.mean()

    return mse


if __name__ == "__main__":
    forecast_batch = Batch(
        surf_vars={k: torch.randn(1, 2, 17, 32) for k in ("2t", "10u", "10v", "msl", "tp")},
        static_vars={k: torch.randn(17, 32) for k in ("lsm", "z", "slt")},
        atmos_vars={k: torch.randn(1, 2, 4, 17, 32) for k in ("z", "u", "v", "t", "q")},
        metadata=Metadata(
            lat=torch.linspace(90, -90, 17),
            lon=torch.linspace(0, 360, 32 + 1)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=(100, 250, 500, 850),
        ),
    )
    
    truth_batch = Batch(
        surf_vars={k: forecast_batch.surf_vars[k] + 0.1 * torch.randn_like(forecast_batch.surf_vars[k]) 
                  for k in forecast_batch.surf_vars.keys()},
        static_vars=forecast_batch.static_vars,
        atmos_vars={k: forecast_batch.atmos_vars[k] + 0.1 * torch.randn_like(forecast_batch.atmos_vars[k])
                   for k in forecast_batch.atmos_vars.keys()},
        metadata=forecast_batch.metadata,
    )
    
    rmse_2t = torch.sqrt(mse(forecast_batch, truth_batch, "2t"))
    print(f"RMSE for 2t: {rmse_2t:.4f}")
    
    rmse_z_500 = torch.sqrt(mse(forecast_batch, truth_batch, "z", 500))
    print(f"RMSE for z at 500 hPa: {rmse_z_500:.4f}")
    
    rmse_2t = torch.sqrt(mse(forecast_batch, truth_batch, "2t", latitude_weights=True))
    print(f"RMSE with latitude weight for 2t: {rmse_2t:.4f}")
    
    rmse_z_500 = torch.sqrt(mse(forecast_batch, truth_batch, "z", 500, latitude_weights=True))
    print(f"RMSE with latitude weight for z at 500 hPa: {rmse_z_500:.4f}")
