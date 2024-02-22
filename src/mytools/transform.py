import xarray as xr


def roll(ds: xr.Dataset, xdim: str = "lon") -> xr.Dataset:
    """Rolls data to -180:180 longitude format."""
    # TODO implement the reverse of this too
    # TODO or make it arbitrary by giving a center line?
    ds_east = ds.sel(**{xdim: slice(180, 360)})
    ds_east = ds_east.assign_coords(**{xdim: (ds_east[xdim] - 360)})
    ds_west = ds.sel({xdim: slice(0, 180)})
    return xr.concat([ds_east, ds_west], dim=xdim)
