#!/usr/bin/env python
'''
Classify AVHRR GAC data surface types and compute sea ice concentration

Surface classification code written by Steinar Eastwood, FoU

'''

import os
import h5py
import argparse
import datetime

import numpy as np
import numpy.ma as ma
import pyresample as pr
import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
from scipy import stats
from pyresample import kd_tree
import pyresample as pr

import netCDF4
import datetime

from matplotlib import mlab
from scipy import ndimage

def load_extent_mask(filepath):
    '''
    Args:
        filepath (str) : path to filepath
    '''
    data = np.load(filepath)
    extent_mask = data['extent_mask']
    return extent_mask

def solve(m1,m2,std1,std2):
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c  = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    return np.roots([a,b,c])

def cleanup_coefficients(array):
    """
    Mask out zeros because we essentially have no data there
    and mask out NaNs
    """
    array_fixed = np.ma.fix_invalid(array)
    array_ma = np.ma.array(array, mask = array_fixed.mask + (array_fixed == 0))

    angles = array_ma[:,0]
    mean_c = array_ma[:,1]
    mean_s = array_ma[:,2]
    counts = array_ma[:,3]
    weights = (counts-counts.min())/(counts.max()-counts.min())

    mask = mean_c.mask + (mean_s>=mean_c)
    angles_ma = angles[~mask]
    mean_c_ma = mean_c[~mask]
    mean_s_ma = mean_s[~mask]
    counts_ma = counts[~mask]
    weights_ma = weights[~mask]

    return angles_ma, mean_c_ma, mean_s_ma, weights_ma

def clean_up_cloudmask(cloudmask, lats):
    """ Clean up large chunks of errors in cloudmask

    Return updated cloudmask

    There appears to be big blocks of open water in the cloudmask around the pole
    Which is clearly wrong.

    Use scipy.ndimage to find large objects and remove them
    Solution adopted from "http://www.scipy-lectures.org/advanced/image_processing"
    """
    mask = np.where(((cloudmask==1) * (lats>80))==True, True, False)
    labels, nlabels = ndimage.label(mask)
    sizes = ndimage.sum(mask, labels, range(nlabels + 1))
    mask_sizes = sizes > 50
    remove_pixels = mask_sizes[labels]
    updated_cloudmask_data = np.where(remove_pixels==True, 0, cloudmask.data)

    return np.ma.array(updated_cloudmask_data,  mask= cloudmask.mask)


def compute_sic( data1, data2, tb11, cloudmask, coeffs, lons, lats, soz, sez, cloudprob ):
    """compute sea ice concentration

    use probability information to select tie points
    pixel with highest probability of sea ice or water
    are selected as dynamic tiepoints

    args:
        data (numpy.ndarray):   observation array for computing sic

    returns:
        sic (numpy.ndarray):    array with sea ice concentration values
    """

    sic_with_water = np.ma.array(np.zeros(cloudmask.shape), mask=cloudmask.mask)



    angles, means, stds, weights = cleanup_coefficients(coeffs)

    from scipy.interpolate import UnivariateSpline, interp1d
    spline_means = UnivariateSpline(angles, means, w=weights)
    spline_stds  = UnivariateSpline(angles, stds, w=weights)
    ice_mean = np.ma.array(spline_means(soz.ravel()).reshape(soz.shape), mask = soz.mask)
    ice_std = np.ma.array(spline_stds(soz.ravel()).reshape(soz.shape), mask = soz.mask)

    # pick the pixels where the probability of ice is higher than other surface types
    # don't use pixels that are: clouds contaminated (2)
    #                            clouds filled (3)
    #                            not processed (0)
    #                            undefined (5)
    #                            sun elevation angles above 90 degrees
    water_threshold = 9 # reflectance of water is roughly 3 percent
    water_mask = (cloudmask == 2) * (data1 < (water_threshold + 1))

    sic = 100.*(data1 - water_threshold) /(ice_mean - 0.5*ice_std - water_threshold)
    sic = np.where((water_mask == True), 0, sic)

    mask = ((cloudmask != 4) * (cloudmask != 2)) + \
            (soz > 80) + \
            (soz < 50) + \
            (sez > 55) + \
            (cloudprob > 5)

    sic = np.ma.array(sic, mask = (mask + (data1>90)) )
    re0609 = np.ma.array(data1/data2, mask = sic.mask + (sic <15 ))

    return sic, re0609

def get_osisaf_land_mask(filepath):
    """
    Load a OSI SAF landmask using numpy.load
    args:
        filepath (str) : path to file
    """
    data = np.load(filepath)
    land_mask = data['land_mask'].astype('bool')
    return land_mask

def save_sic(output_filename, sic, re0609, timestamp, lon, lat):

    filehandle = netCDF4.Dataset(output_filename, 'w')
    filehandle.createDimension('time', size=1)
    filehandle.createVariable('time', 'l', dimensions=('time'))
    filehandle.variables['time'].units = "seconds since 1970-1-1"
    filehandle.variables['time'][:] = timestamp
    filehandle.createDimension('x', size=sic.shape[0])
    filehandle.createDimension('y', size=sic.shape[1])

    # filehandle.createVariable('lon', 'float', dimensions=( 'x', 'y'))
    # filehandle.createVariable('lat', 'float', dimensions=('x', 'y'))
    # filehandle.variables['lon'].units = 'degrees_east'
    # filehandle.variables['lat'].units = 'degrees_north'
    # filehandle.variables['lon'][:] = lon
    # filehandle.variables['lat'][:] = lat
    # filehandle.variables['lon'].missing_value = -32767
    # filehandle.variables['lon'].fill_value = -32767
    # filehandle.variables['lat'].missing_value = -32767
    # filehandle.variables['lat'].fill_value = -32767

    filehandle.createVariable('ice_conc', 'f4', dimensions=('time', 'x', 'y'), zlib=True,least_significant_digit=2)
    filehandle.variables['ice_conc'].coordinates = "lon lat"
    filehandle.variables['ice_conc'].units = "%"
    filehandle.variables['ice_conc'].missing_value = -32767
    filehandle.variables['ice_conc'].fill_value = -32767
    filehandle.variables['ice_conc'].valid_min = 0.
    filehandle.variables['ice_conc'].valid_max = 100.
    filehandle.variables['ice_conc'][:] = sic

    filehandle.createVariable('re0609', 'f4', dimensions=('time', 'x', 'y'), zlib=True,least_significant_digit=2)
    filehandle.variables['re0609'].coordinates = "lon lat"
    filehandle.variables['re0609'].units = "na"
    filehandle.variables['re0609'].missing_value = -32767
    filehandle.variables['re0609'].fill_value = -32767
    filehandle.variables['re0609'].valid_min = 0.
    filehandle.variables['re0609'].valid_max = 10.
    filehandle.variables['re0609'][:] = re0609



    filehandle.close()

def compose_filename(data, sensor_name):
    timestamp = datetime.datetime.fromtimestamp(data.variables['time'][0])
    timestamp_string = timestamp.strftime('%Y%m%d_%H%M')
    filename = '{}_iceconc_{}_arctic.nc'.format(sensor_name,timestamp_string)
    return filename

def apply_mask(mask_array, data_array):
    """
    Apply mask to data array

    Args:
        mask (numpy.ndarray) : boolean array
        data (numpy.ma.ndarray) : numerical masked array

    Returns:
        masked_data_array (numpy.ma.ndarray) : masked array
    """
    # masked_data_array = np.ma.array(data_array.data, mask = data_array.mask + mask_array)

    masked_data_array = np.where(mask_array == True, 200, data_array.data)
    original_mask = data_array.mask

    combined_mask = np.where(mask_array == True, False, original_mask)
    data_array_with_combined_mask = np.ma.array(masked_data_array, mask = combined_mask)

    return data_array_with_combined_mask


def main():

    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output-dir", default='.', nargs=1)
    p.add_argument('-c', '--coeffs', nargs=1,
                         help='Name of the area definition',
                         type=str)
    p.add_argument("-i", "--input-file", nargs=1,
                         help="Input Mitiff Files")
    p.add_argument("-a", "--areas_file", nargs=1,
                         help="Areas definition file")
    p.add_argument('-s', '--sensor', nargs=1,
                         help='Name of the sensor, e.g. avhrr_metop02',
                         type=str)
    p.add_argument('-m', '--mean-coeffs', nargs=1, help='mean and standard deviation over ice')
    p.add_argument('-e', '--extent-mask-file', nargs=1, help='climatological ice extent mask')

    args = p.parse_args()
    areas_filepath = args.areas_file[0]

    sensor_name = args.sensor[0]
    mean_coeffs = np.load(args.mean_coeffs[0])

    # Read in test AVHRR swath file, with lat/lon info (for trimming)
    avhrr_filepath = args.input_file[0]
    avhrr_dirpath = os.path.dirname(avhrr_filepath)
    avhrr_basename = os.path.basename(avhrr_filepath)
    avhrr = netCDF4.Dataset(avhrr_filepath, locations=True)

    vis06 = avhrr.variables['vis06'][0,:,:]
    vis09 = avhrr.variables['vis09'][0,:,:]
    tb11  = avhrr.variables['tb11'][0,:,:]
    lons = avhrr.variables['lon'][0,:,:]
    lats = avhrr.variables['lat'][0,:,:]
    cloudmask = avhrr.variables['cloudmask'][0,:,:]
    cloudprob = avhrr.variables['cloudprob'][0,:,:]

    soz = avhrr.variables['sunsatangles'][0,:,:]
    sez = avhrr.variables['sensorangles'][0,:,:]

    cloudmask = clean_up_cloudmask(cloudmask, lats)
    sic, re0609 = compute_sic(vis06, vis09, tb11, cloudmask, mean_coeffs, lons, lats, soz, sez, cloudprob)

    sic_filename = compose_filename(avhrr, sensor_name)
    output_path = os.path.join(args.output_dir[0], sic_filename)

    # Load OSI SAF landmask and apply to resampled SIC array
    land_mask_filepath = os.path.join(os.path.dirname(
                                      os.path.abspath(__file__)),
		                              'resources',
                                      'land_mask_4k.npz')

    land_mask = get_osisaf_land_mask(land_mask_filepath)

    save_sic(output_path,
                 sic,
                 re0609,
                 avhrr.variables['time'][0],
                 avhrr.variables['lon'][:,:],
                 avhrr.variables['lat'][:,:])


if __name__ == '__main__':
    main()
