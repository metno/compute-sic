#!/usr/bin/env python
'''
Compute mean and std of selected AVHRR gac channels distributed by solar zenith angle
'''
import argparse
import os
import numpy as np
import netCDF4 as nc
from scipy import stats
from matplotlib import pyplot as plt
import pyresample as pr

def load_data(input_filename, varname):
    """
    Load resampled AVHRR GAC data stored in netcdf format
    Args:
        varname (str) : name of the variable to load
        input_filename (str) : path to input netcdf file
    Returns:
        data (np.ndarray) : observations array
        angles (np.ndarray) : sun elevation angles array
        latitudes (np.ndarray) : latitudes mask
    """
    dataset = nc.Dataset(input_filename)
    data = dataset.variables[varname][:]
    angles = dataset.variables['sunsatangles'][:]
    latitudes = dataset.variables['lat'][:]
    cloudmask = dataset.variables['cloudmask'][:]
    return data, angles, latitudes, cloudmask


def load_land_mask(filepath):
    '''
    Args:
        filepath (str) : path to filepath
    '''
    data = np.load(filepath)
    land_mask = data['land_mask'].astype('bool')
    return land_mask


def load_extent_mask(filepath):
    '''
    Args:
        filepath (str) : path to filepath
    '''
    data = np.load(filepath)
    extent_mask = data['extent_mask']
    return extent_mask


def compress_data(data, sunsatangles, mask=None):
    '''
    Compute mean and standard deviation of `data` sorted by sunsatangles
    Args:
        data (np.ndarray) : data to be processed
        mask (np.ndarray) : mask
    '''
    angles_ma = np.ma.array(sunsatangles, mask = ~mask).compressed()
    data_ma = np.ma.array(data, mask = ~mask).compressed()
    return data_ma, angles_ma


def main():

    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output-dir", nargs=1)
    p.add_argument("-e", "--extent-mask", default='extent.npy', nargs=1)
    p.add_argument("-l", "--land-mask", default='landmask.npy', nargs=1)
    p.add_argument("input_files", nargs='*')
    p.add_argument('-s', '--sensor', nargs=1,
                         help='Name of the sensor, e.g. avhrr_metop02',
                         type=str)
    p.add_argument("-a", "--areas_file", nargs=1,
                         help="Areas definition file")

    args = p.parse_args()
    output_dir = args.output_dir[0]
    input_files = args.input_files
    land_mask_file = args.land_mask[0]
    extent_mask_file = args.extent_mask[0]
    areas_filepath = args.areas_file[0]

    area_def = pr.utils.load_area(areas_filepath, 'nsidc_stere_north_10k')
    (area_def.lons, area_def.lats) = area_def.get_lonlats()


    output_filename = '{}-coeffs'.format(args.sensor[0])
    output_path = os.path.join(output_dir, output_filename)

    land_mask = load_land_mask(land_mask_file)
    extent_mask = load_extent_mask(extent_mask_file)

    data_ma = np.array([])
    angles_ma = np.array([])
    for input_file in input_files:
        print input_file
        (data, angles, latitudes, cloudmask) = load_data(input_file, 'vis09')

        # use only data within the extent and the surface type is 4 (sea ice)
        mask = (extent_mask == True) * (cloudmask == 4) * (land_mask == False)
        data_co, angles_co = compress_data(data, angles, mask=mask)
        data_ma = np.append(data_ma, data_co); angles_ma = np.append(angles_ma, angles_co)

    coeffs = np.zeros((90,4))
    # for i in range(int(angles_ma.min()),int(angles_ma.max())):
    for i in range(int(angles_ma.min()), 90):
        refl = data_ma[angles_ma.astype(np.int)==i]
        try:
            coeffs[i,:] = ([i, np.median(refl), refl.std(), refl.shape[0]])
        except:
            coeffs[i,:] = (0,0,0,0)

    np.save(output_path, coeffs)


if __name__ == "__main__":
    main()
