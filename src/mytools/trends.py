from typing import Tuple

import numpy as np
import xarray as xr


def remove_trends(
    da: xr.DataArray,
    start: str,
    end: str,
    time_dim: str = "time",
    method: str = "linear",
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Remove linear trends from a given DataArray within a specified time range.
    The anomaly is referenced to the mean of the time period from `start` to `end`.

    Parameters:
        da (xr.DataArray): The input DataArray.
        start (str): The start date of the time range.
        end (str): The end date of the time range.
        time_dim (str, optional): The name of the time dimension. Defaults to "time".
        method (str, optional): The method used for trend removal. Only "linear" is implemented. Defaults to "linear".

    Returns:
        Tuple[xr.DataArray, xr.DataArray]: A tuple containing the detrended DataArray and the trend component.
    """
    if method.lower() != "linear":
        raise NotImplementedError("Only method=linear is implemented.")
    result = da.sel(**{time_dim: slice(start, end)}).polyfit(
        dim=time_dim, deg=1, skipna=True
    )
    time = xr.DataArray(
        data=np.arange(len(da.time)) * 86_400_000_000_000,
        dims=[time_dim],
        coords=dict(
            time=da.time,
        ),
    )
    trend = result.polyfit_coefficients.sel(degree=1).dot(time)
    mean = (da - trend).sel(**{time_dim: slice(start, end)}).mean(time_dim)
    return (da - mean - trend).drop_vars("degree"), trend - mean
