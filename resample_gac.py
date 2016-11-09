#!/usr/bin/env python
'''
Read AVHRR GAC data, resample to polarstereographic grid and save
as netcdf file

'''

import os
import h5py
import argparse

import numpy as np
import numpy.ma as ma
import pyresample as pr
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
from scipy import stats
from pyresample import kd_tree

from pylab import griddata

import netCDF4
import datetime
import collections


def calibrate_refl(sez, soz, refl, channel=None):
    ch = None
    if channel == 'ch1':
        ch = AvhrrChannel1()
    elif channel == 'ch2':
        ch = AvhrrChannel2()
    try:
        ch.calibrate(sez, soz, refl)
    except:
        raise TypeError
    if ch.refl.any():
        return ch.refl


class AvhrrLut(object):
    def __init__(self):
        self.x = None
        self.y = None
        self.soz = np.arange(35, 80, 5)
        self.sez = np.arange(0,  75, 5)
        self.angles_grid = np.meshgrid(self.sez,
                                       self.soz)
        self.rp = np.row_stack((self.angles_grid[0].ravel(),
                                self.angles_grid[1].ravel()))
        self.coeff_a = np.ones((15,9))
        self.coeff_b = np.ones((15,9))

    def interpolate(self, sez, soz):
        self.gza = griddata(self.rp[0], self.rp[1], self.coeff_a.ravel(), sez, soz, 'nn')
        self.gzb = griddata(self.rp[0], self.rp[1], self.coeff_b.ravel(), sez, soz, 'nn')

    def calibrate(self, sez, soz, refl):
        self.interpolate(sez, soz)
        calibrated_refl = refl*self.gza + self.gzb
        self.refl = calibrated_refl
        return calibrated_refl
        

class AvhrrChannel1(AvhrrLut):
    def __init__(self):
        AvhrrLut.__init__(self)
        self.coeff_a = np.array([[1.083, 1.091, 1.101, 1.115, 1.133, 1.159, 1.198, 1.260, 1.369],
                                 [1.083, 1.091, 1.115, 1.115, 1.134, 1.160, 1.199, 1.261, 1.369],
                                 [1.084, 1.092, 1.102, 1.116, 1.135, 1.161, 1.200, 1.262, 1.370],
                                 [1.086, 1.094, 1.104, 1.118, 1.136, 1.163, 1.202, 1.264, 1.372],
                                 [1.088, 1.096, 1.107, 1.120, 1.139, 1.165, 1.205, 1.376, 1.376],
                                 [1.092, 1.100, 1.110, 1.124, 1.143, 1.169, 1.208, 1.271, 1.380],
                                 [1.097, 1.104, 1.115, 1.129, 1.147, 1.174, 1.213, 1.276, 1.386],
                                 [1.103, 1.111, 1.121, 1.135, 1.154, 1.181, 1.220, 1.283, 1.393],
                                 [1.111, 1.118, 1.129, 1.143, 1.162, 1.189, 1.229, 1.292, 1.403],
                                 [1.121, 1.129, 1.140, 1.154, 1.173, 1.200, 1.240, 1.304, 1.416],
                                 [1.135, 1.143, 1.154, 1.168, 1.187, 1.215, 1.255, 1.320, 1.433],
                                 [1.154, 1.162, 1.173, 1.187, 1.207, 1.235, 1.276, 1.342, 1.457],
                                 [1.181, 1.189, 1.200, 1.215, 1.235, 1.263, 1.306, 1.372, 1.490],
                                 [1.220, 1.229, 1.240, 1.255, 1.276, 1.306, 1.349, 1.418, 1.539],
                                 [1.283, 1.292, 1.304, 1.320, 1.342, 1.372, 1.418, 1.490, 1.617]])

        self.coeff_b = np.array([[-0.018, -0.017, -0.017, -0.019, -0.021, -0.026, -0.032, -0.043, -0.061],
                                 [-0.021, -0.021, -0.020, -0.021, -0.023, -0.027, -0.034, -0.044, -0.063],
                                 [-0.023, -0.024, -0.024, -0.024, -0.026, -0.030, -0.036, -0.047, -0.065],
                                 [-0.025, -0.026, -0.028, -0.029, -0.030, -0.034, -0.040, -0.050, -0.069],
                                 [-0.028, -0.029, -0.031, -0.034, -0.036, -0.039, -0.045, -0.056, -0.076],
                                 [-0.031, -0.033, -0.034, -0.038, -0.043, -0.047, -0.052, -0.063, -0.084],
                                 [-0.036, -0.037, -0.039, -0.043, -0.048, -0.055, -0.063, -0.073, -0.095],
                                 [-0.043, -0.043, -0.045, -0.049, -0.054, -0.062, -0.074, -0.089, -0.111],
                                 [-0.043, -0.052, -0.052, -0.057, -0.063, -0.071, -0.105, -0.105, -0.134],
                                 [-0.045, -0.052, -0.064, -0.066, -0.073, -0.083, -0.097, -0.121, -0.161],
                                 [-0.049, -0.057, -0.066, -0.082, -0.087, -0.098, -0.115, -0.141, -0.188],
                                 [-0.054, -0.063, -0.073, -0.087, -0.108, -0.118, -0.138, -0.170, -0.222],
                                 [-0.062, -0.071, -0.083, -0.098, -0.118, -0.149, -0.169, -0.208, -0.273],
                                 [-0.074, -0.085, -0.097, -0.115, -0.138, -0.169, -0.220, -0.262, -0.344],
                                 [-0.088, -0.105, -0.121, -0.141, -0.170, -0.208, -0.261, -0.353, -0.449]])


class AvhrrChannel2(AvhrrLut):
    def __init__(self):
        AvhrrLut.__init__(self)
        self.coeff_a = np.array([[1.121, 1.127, 1.135, 1.146, 1.160, 1.180, 1.210, 1.257, 1.337],
                                 [1.122, 1.128, 1.136, 1.146, 1.160, 1.181, 1.210, 1.257, 1.337],
                                 [1.122, 1.128, 1.136, 1.147, 1.161, 1.181, 1.211, 1.258, 1.338],
                                 [1.124, 1.130, 1.138, 1.148, 1.163, 1.183, 1.212, 1.259, 1.339],
                                 [1.126, 1.132, 1.140, 1.150, 1.164, 1.185, 1.214, 1.261, 1.341],
                                 [1.128, 1.134, 1.142, 1.153, 1.167, 1.187, 1.217, 1.263, 1.344],
                                 [1.132, 1.138, 1.146, 1.156, 1.170, 1.191, 1.220, 1.267, 1.347],
                                 [1.136, 1.142, 1.150, 1.161, 1.175, 1.195, 1.225, 1.272, 1.352],
                                 [1.142, 1.148, 1.156, 1.167, 1.181, 1.201, 1.231, 1.278, 1.358],
                                 [1.150, 1.156, 1.164, 1.174, 1.189, 1.209, 1.239, 1.285, 1.367],
                                 [1.161, 1.167, 1.174, 1.185, 1.199, 1.219, 1.249, 1.296, 1.378],
                                 [1.175, 1.181, 1.189, 1.199, 1.214, 1.234, 1.264, 1.311, 1.393],
                                 [1.195, 1.201, 1.209, 1.219, 1.234, 1.254, 1.284, 1.332, 1.414],
                                 [1.225, 1.231, 1.239, 1.249, 1.264, 1.284, 1.315, 1.363, 1.446],
                                 [1.272, 1.278, 1.285, 1.296, 1.311, 1.332, 1.363, 1.412, 1.497]])

        self.coeff_b = np.array([[-0.0125, -0.011, -0.011, -0.012, -0.013, -0.015 , -0.0187, -0.025, -0.036],
                                 [-0.0145, -0.014, -0.013, -0.013, -0.014, -0.0158, -0.0192, -0.025, -0.036],
                                 [-0.0156, -0.016, -0.016, -0.015, -0.016, -0.0171, -0.0204, -0.026, -0.036],
                                 [-0.0169, -0.018, -0.019, -0.019, -0.018, -0.0194, -0.0223, -1.259, -0.038],
                                 [-0.0191, -0.019, -0.02 , -0.022, -0.023, -0.023 , -0.0253, -0.031, -0.041],
                                 [-0.0211, -0.022, -0.023, -0.024, -0.027, -0.0283, -0.0301, -0.035, -0.046],
                                 [-0.0205, -0.022, -0.023, -0.024, -0.026, -0.0284, -0.0305, -0.033, -0.039],
                                 [-0.0292, -0.027, -0.029, -0.031, -0.034, -0.0381, -0.0445, -0.052, -0.063],
                                 [-0.0271, -0.035, -0.033, -0.036, -0.039, -0.0434, -0.0508, -0.062, -0.078],
                                 [-0.0292, -0.033, -0.042, -0.041, -0.045, -0.0513, -0.0586, -0.072, -0.095],
                                 [-0.0314, -0.036, -0.041, -0.053, -0.052, -0.0601, -0.0702, -0.084, -0.111],
                                 [-0.037 , -0.039, -0.045, -0.052, -0.068, -0.0704, -0.0837, -0.103, -0.132],
                                 [-0.0381, -0.043, -0.051, -0.06 , -0.071, -0.0933, -0.1004, -0.125, -0.165],
                                 [-0.0445, -0.051, -0.059, -0.07 , -0.084, -0.1004, -0.1362, -0.155, -0.207],
                                 [-0.0515, -0.062, -0.072, -0.084, -0.103, -0.1252, -0.1547, -0.218, -0.267]])

def save_netcdf(output_path,
                variables=None,
                timestamp = None,
                lons=None,
                lats=None,
                area_def=None,
                resample=True):



    # split pass in two pieces as pyresample is not able to give priority to the
    # most recent data over the old one. That causes problem when the same pass
    # covers the same area twice
    y_size = lons.shape[0]
    y_halfsize = y_size/2

    # strip 20 lines from the beginning and the end as they are often corrupt
    idxs = [20, y_halfsize, y_size-20]

    for i, idx in enumerate(idxs[:-1]):

        idx_1 = idxs[i]
        idx_2 = idxs[i+1]

        swath_def = pr.geometry.SwathDefinition(lons=lons[idx_1:idx_2,:], lats=lats[idx_1:idx_2,:])

        valid_input_index, valid_output_index, index_array, distance_array = \
                                kd_tree.get_neighbour_info(swath_def,
                                                          area_def, 15000,
                                                           neighbours=1, nprocs=8)

        for dataset_name in variables.keys():

            if variables[dataset_name]['data'] is not None:
                dataset = variables[dataset_name]['data'][idx_1:idx_2,:]
                dataset_res = kd_tree.get_sample_from_neighbour_info('nn',
                                                        area_def.shape, dataset.astype(np.float),
                                                        valid_input_index, valid_output_index,
                                                        index_array, fill_value=-32767)

                if variables[dataset_name]['dataset_res'] is None:
                    variables[dataset_name]['dataset_res'] = dataset_res
                else:
                    variables[dataset_name]['dataset_res'] = np.where(dataset_res != -32767, dataset_res, variables[dataset_name]['dataset_res'])

    # Once the datasets have been resampled save them into netcdf file
    # create netcdf file
    filehandle = netCDF4.Dataset(output_path, 'w')
    filehandle.createDimension('time', size=1)
    filehandle.createVariable('time', 'l', dimensions=('time'))
    filehandle.variables['time'].units = "seconds since 1970-1-1"
    filehandle.variables['time'][:] = timestamp
    filehandle.createDimension('x', size=area_def.lons.shape[0])
    filehandle.createDimension('y', size=area_def.lons.shape[1])

    for dataset_name in variables.keys():
        filehandle.createVariable(dataset_name, 'f4', dimensions=('time', 'x', 'y'))
        filehandle.variables[dataset_name].coordinates = "lon lat"
        filehandle.variables[dataset_name].units = variables[dataset_name]['units']
        filehandle.variables[dataset_name].missing_value = -32767
        filehandle.variables[dataset_name].fill_value = -32767
        filehandle.variables[dataset_name][0,:,:] = variables[dataset_name]['dataset_res']
    filehandle.close()

def compose_filename(data, sensor_name):
    timestamp = data.timestamp
    timestamp_string = timestamp.strftime('%Y%m%d_%H%M')
    filename = 'avhrr-gac_{}_{}_arctic.nc'.format(sensor_name, timestamp_string)
    return filename


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

    # p.add_argument("-s", "--satellite_name")
    # p.add_argument('-b', '--channels', nargs='+',
    #                      help='Input channels',
    #                      type=str)
    args = p.parse_args()
    areas_filepath = args.areas_file[0]

    ''' Test wic-classifier in python.'''

    sensor_name = args.sensor[0]

    # Read in test AVHRR swath file, with lat/lon info (for trimming)
    avhrr_filepath = args.input_file[0]
    avhrr_dirpath = os.path.dirname(avhrr_filepath)
    avhrr_basename = os.path.basename(avhrr_filepath)
    avhrr = AvhrrData(avhrr_filepath, locations=True)

    output_filename = compose_filename(avhrr, sensor_name)

    angles_basename = avhrr_basename.replace('avhrr', 'sunsatangles')
    angles_filepath = os.path.join(avhrr_dirpath, angles_basename)
    angles = AngleData(angles_filepath)

    cloudmask_basename = avhrr_basename.replace('avhrr', 'CMA')
    cloudmask_filepath = os.path.join(avhrr_dirpath, cloudmask_basename)
    cloudmask = CloudMaskData(cloudmask_filepath)


    # swath_def = pr.geometry.SwathDefinition(lons=avhrr.lon, lats=avhrr.lat)
    area_def = pr.utils.load_area(areas_filepath, 'nsidc_stere_north_10k')
    (area_def.lons, area_def.lats) = area_def.get_lonlats()

    refl1 = avhrr.data[1]
    refl2 = avhrr.data[2]
    #refl1 = calibrate_refl(angles.data[2], angles.data[1], avhrr.data[1], channel='ch1')
    #refl2 = calibrate_refl(angles.data[2], angles.data[1], avhrr.data[2], channel='ch2')

    variables_dict = collections.OrderedDict()
    variables_dict['vis06'] = { 'name': 'Reflectance 0.6', 'data': refl1, 'units': '', 'dataset_res': None }
    variables_dict['vis09'] = { 'name': 'Reflectance 0.9', 'data': refl2, 'units': '', 'dataset_res': None }
    variables_dict['tb37'] = { 'name': 'Brightness temperature 2.7', 'data': avhrr.data[3], 'units': 'K', 'dataset_res': None }
    variables_dict['tb11'] = { 'name': 'Brightness temperature 11', 'data': avhrr.data[4], 'units': 'K', 'dataset_res': None}
    variables_dict['tb12'] = { 'name': 'Brightness temperature 12', 'data': avhrr.data[5], 'units': 'K' , 'dataset_res': None}
    variables_dict['vis16'] = { 'name': 'Reflectance 0.6', 'data': avhrr.data[6], 'units': '' , 'dataset_res': None}
    variables_dict['cloudmask'] = { 'name': 'Categorical cloudmask', 'data': cloudmask.data, 'units': '' , 'dataset_res': None}
    variables_dict['sunsatangles'] = { 'name': 'Sun elevation angles', 'data': angles.data[1], 'units': 'degrees', 'dataset_res': None}
    variables_dict['sensorangles'] = { 'name': 'Sensor zenith angles', 'data': angles.data[2], 'units': 'degrees', 'dataset_res': None}
    variables_dict['lon'] = { 'name': 'Longitudes', 'data': avhrr.lon, 'units': 'degrees_east', 'dataset_res': None}
    variables_dict['lat'] = { 'name': 'Latitudes', 'data': avhrr.lat, 'units': 'degrees_north', 'dataset_res': None }

    netcdf_filename = compose_filename(avhrr, sensor_name)
    output_path = os.path.join(args.output_dir[0], netcdf_filename)

    timestamp = (avhrr.timestamp - datetime.datetime(1970,1,1)).total_seconds()

    save_netcdf(output_path,
                variables = variables_dict,
                timestamp = timestamp,
                lons = avhrr.lon,
                lats = avhrr.lat,
                area_def = area_def,
                resample=True)


class AvhrrData(object):
    """ Container for AVHRR GAC swath data.

    Channels accessed by: data[ch], where ch are [_, A1, A2, T3, T4, T5, A3]
    """

    pre = ['pass', 'A1', 'A2', 'T3', 'T4', 'T5', 'A3']

    def __init__(self, filename, locations=False):
        """ Read in AVHRR data file in HDF5 format and cleans it up for ready use."""

        self._filehandle = h5py.File(filename, 'r')
        avhrr = self._filehandle
        self.data, self.nodata, self.missingdata = [[None]*7 for i in range(3)]

        self._get_timestamp()

        # Collect data from all channels to data[ch], including gain and offset on
        # all valid values.
        for ch in range(1,7):
            try:
                image = 'image' + str(ch)
                self.data[ch] = np.float32(avhrr[image]['data'].value)
                offset = avhrr[image]['what'].attrs['offset'] # To check for K?
                gain = avhrr[image]['what'].attrs['gain']
                self.nodata[ch] = avhrr[image]['what'].attrs['nodata']
                self.missingdata[ch] = avhrr[image]['what'].attrs['missingdata']
                mask = (self.data[ch] != self.missingdata[ch]) & (self.data[ch] != self.nodata[ch])
                self.data[ch][mask] = self.data[ch][mask] * gain + offset
            except:
                pass

        if locations:
            lat = avhrr['where']['lat']['data'].value
            lon = avhrr['where']['lon']['data'].value
            gain = avhrr['where']['lat']['what'].attrs['gain']
            self.lat = lat * gain
            self.lon = lon * gain

    def _get_timestamp(self):
        """
        """
        start_epoch = self._filehandle['how'].attrs['startepochs']

        timestamp = datetime.datetime.utcfromtimestamp(float(start_epoch))
        self.timestamp = timestamp


class AngleData(object):
    """ Read in AVHRR angles file in HDF5 format and clean it up for ready use."""
    pre = ['pass', 'soz', 'saz', 'pass', 'azu', 'aza']

    def __init__(self, filename):
        angles = h5py.File(filename, 'r')
        self.data, self.nodata, self.missingdata = [[None]*6 for i in range(3)]
        for ch in range(1,6):
            image = 'image{}'.format(ch)
            self.data[ch] = np.float32(angles[image]['data'].value)
            offset = angles[image]['what'].attrs['offset']  # For K check?
            gain = angles[image]['what'].attrs['gain']
            self.nodata[ch] = angles[image]['what'].attrs['nodata']
            self.missingdata[ch] = angles[image]['what'].attrs['missingdata']

            mask = (self.data[ch] != self.missingdata[ch]) & (self.data[ch] != self.nodata[ch])
            self.data[ch][mask] = self.data[ch][mask] * gain + offset

        angles.close()

    def describe(self):
        print '''image 1: sun zenith angle. \n
                image 2: satellite zenith angle. \n
                image 3: relative sun-satellite azimuth difference angle \n
                image 4: absolute sun azimuth angle \n
                image 5: absolute satellite azimuth angle \n'''


class CloudMaskData(object):
    """
    Read GAC AVHRR CLARA (comes from CM-SAF) cloud masking information

    """
    def __init__(self, filename):
        filehandle = h5py.File(filename, 'r')
        self.data = filehandle['cma_extended'].value
        self.cloudmask_value_namelist = filehandle['cma_extended'].attrs['flag_meanings']


if __name__ == '__main__':
    main()
