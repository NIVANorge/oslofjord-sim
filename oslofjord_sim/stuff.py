import xml.etree.ElementTree as ET

import numpy as np
import requests
import xesmf as xe
from scipy import ndimage
from scipy.interpolate import interp1d

CATALOG_URL = "https://thredds.met.no/thredds/catalog/fou-hi/norkyst800m/catalog.xml"


def tranform_to_z(ds):
    """
    Transforms s coordingate to z with Vtransform = 2
    """
    zo_rho = (ds.hc * ds.s_rho + ds.Cs_r * ds.h) / (ds.hc + ds.h)
    z_rho = ds.zeta + (ds.zeta + ds.h) * zo_rho
    return z_rho.transpose()


def fill_nans_with_nearest(da):
    """Fill NaNs in an xarray DataArray with nearest non-NaN values."""

    # mask: 1 where data is NaN, 0 elsewhere
    nan_mask = np.isnan(da.values)

    # distance transform returns:
    #  - distances
    #  - indices of nearest non-NaN pixel along each axis
    dist, idx = ndimage.distance_transform_edt(nan_mask, return_indices=True)

    # use advanced indexing to build the filled array
    filled = da.values[tuple(idx)]

    return filled


def regrid_depths(values, depths, target_depths):
    """
    Args:
        values: values to be interpolated
        depths: depths of original values
        target_depths: interpolation target depths
    Returns:
        interpolated_values: result values on target depths
    """
    interpolated_shape = list(depths.shape)
    interpolated_shape[1] = len(target_depths)
    interpolated_values = np.empty(interpolated_shape)

    T, D, X, Y = values.shape
    for t in range(T):
        for x in range(X):
            for y in range(Y):
                f = interp1d(
                    depths[t, :, x, y],
                    values[t, :, x, y],
                    kind="linear",
                    bounds_error=False,
                )
                interpolated_values[t, :, x, y] = f(target_depths)

    return interpolated_values


def regrid_from_norkyst(regridder_rho, regridder_u, regridder_v, ds_in, ds_out_c, ds_out_u, ds_out_v, target_depths):
    """
    Regrids norkyst output file from
    https://thredds.met.no/thredds/catalog/fou-hi/norkyst800m/catalog.html
    'ds_in' to target lons, lats, depths.
    Target lons and lats are in ds_out_*, for example

    ds_out_c = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(59.1, 59.98, num=490), {"units": "degrees_north"}),
            "lon": (["lon"], np.linspace(10.2, 10.85, num=88), {"units": "degrees_east"}),
        }
    )
    ds_out_u = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(59.1, 59.98, num=490), {"units": "degrees_north"}),
            "lon": (["lon"], np.linspace(10.2, 10.85, num=88 + 1), {"units": "degrees_east"}),
        }
    )
    ds_out_v = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(59.1, 59.98, num=490 + 1), {"units": "degrees_north"}),
            "lon": (["lon"], np.linspace(10.2, 10.85, num=88), {"units": "degrees_east"}),
        }
    )

    """
    if regridder_rho is None:
        regridder_rho = xe.Regridder(ds_in, ds_out_c, "bilinear", unmapped_to_nan=True)
    if regridder_u is None:
        regridder_u = xe.Regridder(ds_in, ds_out_u, "bilinear", unmapped_to_nan=True)
    if regridder_v is None:
        regridder_v = xe.Regridder(ds_in, ds_out_v, "bilinear", unmapped_to_nan=True)

    da_temp = regridder_rho(ds_in["temperature"])
    da_salt = regridder_rho(ds_in["salinity"])
    da_u = regridder_u(ds_in["u_eastward"])
    da_v = regridder_v(ds_in["v_northward"])

    depths = ds_in.depth.values
    f = interp1d(
        -1 * depths,
        da_temp.values,
        axis=1,
        kind="linear",
        bounds_error=False,
    )
    np_temp = f(target_depths)
    f = interp1d(
        -1 * depths,
        da_salt.values,
        axis=1,
        kind="linear",
        bounds_error=False,
    )
    np_salt = f(target_depths)
    f = interp1d(
        -1 * depths,
        da_u.values,
        axis=1,
        kind="linear",
        bounds_error=False,
    )
    np_u = f(target_depths)
    f = interp1d(
        -1 * depths,
        da_v.values,
        axis=1,
        kind="linear",
        bounds_error=False,
    )
    np_v = f(target_depths)

    np_time = ds_in.time.values

    return regridder_rho, regridder_u, regridder_v, np_time, np_temp, np_salt, np_u, np_v


def regrid_from_roms(ds_in, ds_out_c, ds_out_u, ds_out_v, target_depths):
    """
    Regrids roms output file 'ds_in' to target lons, lats, depths.
    Target lons and lats are in ds_out_*, for example

    ds_out_c = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(59.1, 59.98, num=490), {"units": "degrees_north"}),
            "lon": (["lon"], np.linspace(10.2, 10.85, num=88), {"units": "degrees_east"}),
        }
    )
    ds_out_u = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(59.1, 59.98, num=490), {"units": "degrees_north"}),
            "lon": (["lon"], np.linspace(10.2, 10.85, num=88 + 1), {"units": "degrees_east"}),
        }
    )
    ds_out_v = xr.Dataset(
        {
            "lat": (["lat"], np.linspace(59.1, 59.98, num=490 + 1), {"units": "degrees_north"}),
            "lon": (["lon"], np.linspace(10.2, 10.85, num=88), {"units": "degrees_east"}),
        }
    )

    """
    ds_in["z_rho"] = tranform_to_z(ds_in)

    regridder_rho = xe.Regridder(
        ds_in.rename({"lon_rho": "lon", "lat_rho": "lat"}), ds_out_c, "bilinear", unmapped_to_nan=True
    )
    regridder_u = xe.Regridder(
        ds_in.rename({"lon_u": "lon", "lat_u": "lat"}), ds_out_u, "bilinear", unmapped_to_nan=True
    )
    regridder_v = xe.Regridder(
        ds_in.rename({"lon_v": "lon", "lat_v": "lat"}), ds_out_v, "bilinear", unmapped_to_nan=True
    )

    da_temp = regridder_rho(ds_in["temp"])
    da_salt = regridder_rho(ds_in["salt"])
    da_zrho = regridder_rho(ds_in["z_rho"])
    da_u = regridder_u(ds_in["u"])
    da_v = regridder_v(ds_in["v"])

    zrho = da_zrho.values
    zrho = np.transpose(zrho, (1, 0, 2, 3))

    np_temp = regrid_depths(da_temp.values, zrho, target_depths)
    np_salt = regrid_depths(da_salt.values, zrho, target_depths)

    zu = np.zeros_like(da_u)
    zu[:, :, :, :-1] = zrho
    zu[:, :, :, -1] = zu[:, :, :, -2]
    zv = np.zeros_like(da_v)
    zv[:, :, :-1, :] = zrho
    zv[:, :, -1, :] = zv[:, :, -2, :]

    np_u = regrid_depths(da_u.values, zu, target_depths)
    np_v = regrid_depths(da_v.values, zv, target_depths)

    np_time = ds_in.ocean_time.values

    return np_time, np_temp, np_salt, np_u, np_v


def fill_surrounded_nans(arr):
    # Make a copy to avoid modifying original array
    result = arr.copy()
    rows, cols = arr.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if np.isnan(arr[i, j]):
                west = arr[i, j - 1]
                north = arr[i - 1, j]
                east = arr[i, j + 1]
                south = arr[i + 1, j]

                # Check if all 4 neighbors are NOT NaN
                if not np.isnan(west) and not np.isnan(north) and not np.isnan(east) and not np.isnan(south):
                    result[i, j] = np.mean([west, north, east, south])

    return result


def replace_surrounded_values(arr, sides=3):
    # Create a copy of the array to modify
    new_arr = arr.copy()

    # Get the shape of the array
    rows, cols = arr.shape

    # Iterate through the array (excluding edges to avoid index errors)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if not np.isnan(arr[i, j]):  # Only check non-NaN values
                # Count NaN neighbors
                neighbors = [
                    np.isnan(arr[i - 1, j]) if i > 0 else False,  # Top
                    np.isnan(arr[i + 1, j]) if i < rows - 1 else False,  # Bottom
                    np.isnan(arr[i, j - 1]) if j > 0 else False,  # Left
                    np.isnan(arr[i, j + 1]) if j < cols - 1 else False,  # Right
                ]
                if sum(neighbors) >= sides:
                    new_arr[i, j] = np.nan  # Replace with NaN if surrounded on 3+ sides

    return new_arr


def fill_diagonal_pairs(arr):
    a = arr.copy()

    # Extract all 2×2 blocks using slicing
    tl = a[:-1, :-1]  # top-left
    tr = a[:-1, 1:]  # top-right
    bl = a[1:, :-1]  # bottom-left
    br = a[1:, 1:]  # bottom-right

    # Condition:
    # TL and BR are floats, TR and BL are NaN
    cond = (~np.isnan(tl)) & (~np.isnan(br)) & (np.isnan(tr)) & (np.isnan(bl))

    # Compute fill value
    fill_val = (tl + br) / 2.0

    # Apply to the upper-right cell of each 2×2 block
    a[:-1, 1:][cond] = fill_val[cond]

    return a


def fill_secondary_diagonal_pairs(arr):
    a = arr.copy()

    # Extract all 2×2 slices
    tl = a[:-1, :-1]  # top-left
    tr = a[:-1, 1:]  # top-right
    bl = a[1:, :-1]  # bottom-left
    br = a[1:, 1:]  # bottom-right

    # Condition for secondary diagonal:
    # TR and BL are floats, TL and BR are NaN
    cond = (~np.isnan(tr)) & (~np.isnan(bl)) & (np.isnan(tl)) & (np.isnan(br))

    # Value to insert = average of (TR, BL)
    fill_val = (tr + bl) / 2.0

    # Fill the TOP-LEFT cell of each matching block
    a[:-1, :-1][cond] = fill_val[cond]

    return a


def replace_surrounded_and_clusters(arr, cluster=1):
    new_arr = arr.copy()
    rows, cols = arr.shape

    # First pass: Replace values surrounded by NaNs on at least 3 sides
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if not np.isnan(arr[i, j]):
                # Check top, bottom, left, right
                neighbors = [
                    np.isnan(arr[i - 1, j]) if i > 0 else False,  # Top
                    np.isnan(arr[i + 1, j]) if i < rows - 1 else False,  # Bottom
                    np.isnan(arr[i, j - 1]) if j > 0 else False,  # Left
                    np.isnan(arr[i, j + 1]) if j < cols - 1 else False,  # Right
                ]
                if sum(neighbors) >= 3:
                    new_arr[i, j] = np.nan  # Replace with NaN if surrounded on 3+ sides

    # Second pass: Replace small clusters (≤3 consecutive values) surrounded by NaNs
    def check_and_replace_clusters(arr, axis):
        """Find and replace small clusters of non-NaNs surrounded by NaNs along the given axis."""
        arr = arr.T if axis == 0 else arr  # Transpose if checking vertically

        for i in range(arr.shape[0]):  # Iterate through rows (or columns if transposed)
            row = arr[i]
            nan_mask = np.isnan(row)
            j = 0

            while j < len(row):
                # Find the start of a cluster of non-NaNs
                if not nan_mask[j]:
                    start = j
                    while j < len(row) and not nan_mask[j]:
                        j += 1
                    end = j  # End of cluster (exclusive)

                    # If cluster is 3 or fewer elements and surrounded by NaNs, replace with NaNs
                    if (end - start) <= cluster:
                        left_nan = start == 0 or nan_mask[start - 1]
                        right_nan = end == len(row) or nan_mask[end]
                        if left_nan and right_nan:
                            row[start:end] = np.nan  # Replace the cluster

                j += 1  # Move to the next element

            arr[i] = row  # Update the row in the array

        return arr.T if axis == 0 else arr  # Transpose back if needed

    new_arr = check_and_replace_clusters(new_arr, axis=1)  # Horizontal check
    new_arr = check_and_replace_clusters(new_arr, axis=0)  # Vertical check

    return new_arr


def list_opendap_files(base_url=CATALOG_URL):
    response = requests.get(base_url)

    if response.status_code != 200:
        print("Failed to retrieve the directory listing.")
        return []

    root = ET.fromstring(response.content)
    files = [
        elem.attrib["name"]
        for elem in root.findall(".//{http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0}dataset")
        if elem.attrib["name"].endswith(".nc")
    ]

    return files
