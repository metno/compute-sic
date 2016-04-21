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
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
from scipy import stats
from pyresample import kd_tree

import netCDF4
import datetime
import collections

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
                                                          area_def, 25000,
                                                           neighbours=1, nprocs=8)

        for dataset_name in variables.keys():

            if variables[dataset_name]['data'] is not None:
                dataset = variables[dataset_name]['data'][idx_1:idx_2,:]
                dataset_res = kd_tree.get_sample_from_neighbour_info('nn',
                                                        area_def.shape, dataset,
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

    cloudmask_basename = avhrr_basename.replace('avhrr', 'cloudmask')
    cloudmask_filepath = os.path.join(avhrr_dirpath, cloudmask_basename)
    cloudmask = CloudMaskData(cloudmask_filepath)

    swath_def = pr.geometry.SwathDefinition(lons=avhrr.lon, lats=avhrr.lat)
    area_def = pr.utils.load_area(areas_filepath, 'nsidc_stere_north_10k')
    (area_def.lons, area_def.lats) = area_def.get_lonlats()

    variables_dict = collections.OrderedDict()
    variables_dict['vis06'] = { 'name': 'Reflectance 0.6', 'data': avhrr.data[1], 'units': '', 'dataset_res': None }
    variables_dict['vis09'] = { 'name': 'Reflectance 0.9', 'data': avhrr.data[2], 'units': '', 'dataset_res': None }
    variables_dict['tb37'] = { 'name': 'Brightness temperature 2.7', 'data': avhrr.data[3], 'units': 'K', 'dataset_res': None }
    variables_dict['tb11'] = { 'name': 'Brightness temperature 11', 'data': avhrr.data[4], 'units': 'K', 'dataset_res': None}
    variables_dict['tb12'] = { 'name': 'Brightness temperature 12', 'data': avhrr.data[5], 'units': 'K' , 'dataset_res': None}
    variables_dict['vis16'] = { 'name': 'Reflectance 0.6', 'data': avhrr.data[6], 'units': '' , 'dataset_res': None}
    variables_dict['cloudmask'] = { 'name': 'Categorical cloudmask', 'data': cloudmask.data, 'units': '' , 'dataset_res': None}
    variables_dict['sunsatangles'] = { 'name': 'Sun elevation angles', 'data': angles.data[1], 'units': 'degrees', 'dataset_res': None}
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

        # avhrr.close()


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
        self.data = filehandle['cloudmask'].value
        self.cloudmask_value_namelist = filehandle['cloudmask'].attrs['output_value_namelist']


if __name__ == '__main__':
    main()
