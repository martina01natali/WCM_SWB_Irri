"""
Generic file I/O and data-integrity helpers.

Author:  Martina Natali <martinanatali@cnr.it> (GitHub: martina01natali)
Created: 2026-09-23
License: GPL-3.0 (see LICENSE)
"""
import json
import os

import xarray as xr


def read_json(chosen_file_path: str):
    """Read and return the contents of a JSON file."""
    data = 0
    if os.path.isfile(chosen_file_path):
        with open(chosen_file_path, "r") as file:
            data = json.load(file)
    else:
        raise FileNotFoundError('File not found.')
    return data


def check_nc_integrity(filename):
    """Checks if an nc can be opened with xarray or netCDF4 and returns 0 (True) or 1 (False)."""
    try:
        ds = xr.open_dataset(filename)
        return 0
    except (KeyError, TypeError, OSError, ValueError):
        pass
    try:
        import netCDF4 as nc
        ds = nc.Dataset(filename)
        return 0
    except OSError:
        print(f'File {filename} cannot be read and may be corrupted. Skipping...')
        return 1


def read_nc_basic(filename):
    """Reads an nc file. To be used after check_nc_integrity."""
    try:
        ds = xr.open_dataset(filename)
        return ds
    except (KeyError, TypeError, OSError, ValueError):
        pass
    try:
        import netCDF4 as nc
        ds = nc.Dataset(filename)
        return ds
    except OSError:
        print(f'File {filename} cannot be read and may be corrupted. Skipping...')
        pass


#############################################################################
# METACODING
#############################################################################

def check_non_unique_index(df, verbose=False):
    """Return True if df's index contains duplicated labels, else False."""
    non_unique_index = df.index.duplicated(keep=False)
    if non_unique_index.any():
        non_unique_labels = df.index[non_unique_index]
        if verbose:
            print("Non-unique index labels:")
            print(non_unique_labels)
        return True
    else:
        return False
