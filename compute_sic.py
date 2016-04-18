#!/usr/bin/env python
'''
Classify AVHRR GAC data surface types and compute sea ice concentration

Surface classification code written by Steinar Eastwood, FoU
Avhrr reading routine - part of `gactools` written by Cristina Luis, FoU


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

import netCDF4
import datetime

from matplotlib import mlab

def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

def compute_sic( data, pice, pwater, pclouds, lons, lats ):
    """compute sea ice concentration

    use probability information to select tie points
    pixel with highest probability of sea ice or water
    are selected as dynamic tiepoints

    args:
        data (numpy.ndarray):   observation array for computing sic
        pice (numpy.ndarray):   probability of sea ice
        pwater (numpy.ndarray): probability of water

    returns:
        sic (numpy.ndarray):    array with sea ice concentration values
    """

    ice_mask = pice >= 0.99
    water_mask = pwater >= 0.99
    lats_mask = lats > 65

    ice_data = ma.array(data, mask = (ice_mask * lats_mask) == False)
    ice_hist = np.histogram(ice_data[ice_data.mask == False], 20)
    # ice_max = np.mean(ice_hist[1]) - define what value should be a maximum ice concentration tie point on the fly

    water_data = ma.array(data, mask = ~(water_mask * lats_mask))
    water_hist = np.histogram(water_data[water_data.mask == False], 20)
    water_max = np.max(water_hist[1])

    # pick the pixels where the probability of ice is higher than other surface types
    only_ice_mask = (pice > pwater) * (pice > pclouds) * (lats > 65)
    only_water_mask = (pwater > pice) * (pwater > pclouds ) * (lats > 65)

    only_ice_data = ma.array(data, mask = ~only_ice_mask)

    # compute regression coefficients
    slope, intercept, r_value, p_value, std_err = stats.linregress((water_max, 20), (0, 100))
    sic = slope * only_ice_data + intercept

    sic = np.where(sic>100, 100, sic)
    sic = np.where(sic<0, 0, sic)

    sic_with_water = np.where(only_water_mask == True, 0, sic.data)
    sic = np.ma.array(np.where(only_ice_mask==True, sic, sic_with_water), mask = ~(only_ice_mask + only_water_mask))

    return sic

def get_osisaf_land_mask(filepath):
    """
    Load a OSI SAF landmask using numpy.load
    args:
        filepath (str) : path to file
    """
    data = np.load(filepath)
    land_mask = data['land_mask'].astype('bool')
    return land_mask

def save_sic(output_filename, sic, pice, pwater, pclouds, a06, a09, timestamp, lon, lat):

    filehandle = netCDF4.Dataset(output_filename, 'w')
    filehandle.createDimension('time', size=1)
    filehandle.createVariable('time', 'l', dimensions=('time'))
    filehandle.variables['time'].units = "seconds since 1970-1-1"
    filehandle.variables['time'][:] = timestamp
    filehandle.createDimension('x', size=sic.shape[0])
    filehandle.createDimension('y', size=sic.shape[1])

    filehandle.createVariable('lon', 'float', dimensions=( 'x', 'y'))
    filehandle.createVariable('lat', 'float', dimensions=('x', 'y'))
    filehandle.variables['lon'].units = 'degrees_east'
    filehandle.variables['lat'].units = 'degrees_north'
    filehandle.variables['lon'][:] = lon
    filehandle.variables['lat'][:] = lat
    filehandle.variables['lon'].missing_value = -32767
    filehandle.variables['lon'].fill_value = -32767
    filehandle.variables['lat'].missing_value = -32767
    filehandle.variables['lat'].fill_value = -32767

    filehandle.createVariable('ice_conc', 'f4', dimensions=('time', 'x', 'y'))
    filehandle.variables['ice_conc'].coordinates = "lon lat"
    filehandle.variables['ice_conc'].units = "%"
    filehandle.variables['ice_conc'].missing_value = -32767
    filehandle.variables['ice_conc'].fill_value = -32767
    filehandle.variables['ice_conc'][:] = sic

    # filehandle.createVariable('pice', 'f4', dimensions=('time', 'x', 'y'))
    # filehandle.variables['pice'].coordinates = "lon lat"
    # filehandle.variables['pice'].units = ""
    # filehandle.variables['pice'].missing_value = -32767
    # filehandle.variables['pice'].fill_value = -32767

    # filehandle.createVariable('pwater', 'f4', dimensions=('time', 'x', 'y'))
    # filehandle.variables['pwater'].coordinates = "lon lat"
    # filehandle.variables['pwater'].units = ""
    # filehandle.variables['pwater'].missing_value = -32767
    # filehandle.variables['pwater'].fill_value = -32767

    # filehandle.createVariable('pclouds', 'f4', dimensions=('time', 'x', 'y'))
    # filehandle.variables['pclouds'].coordinates = "lon lat"
    # filehandle.variables['pclouds'].units = ""
    # filehandle.variables['pclouds'].missing_value = -32767
    # filehandle.variables['pclouds'].fill_value = -32767

    # filehandle.createVariable('A06', 'f4', dimensions=('time', 'x', 'y'))
    # filehandle.variables['A06'].coordinates = "lon lat"
    # filehandle.variables['A06'].units = ""
    # filehandle.variables['A06'].missing_value = -32767
    # filehandle.variables['A06'].fill_value = -32767

    # filehandle.createVariable('A09', 'f4', dimensions=('time', 'x', 'y'))
    # filehandle.variables['A09'].coordinates = "lon lat"
    # filehandle.variables['A09'].units = ""
    # filehandle.variables['A09'].missing_value = -32767
    # filehandle.variables['A09'].fill_value = -32767

    # filehandle.variables['avhrr_iceconc'][:] = sic.astype('f4')
    # filehandle.variables['pice'][:] = pice.astype('f4')
    # filehandle.variables['pwater'][:] = pwater.astype('f4')
    # filehandle.variables['pclouds'][:] = pclouds.astype('f4')
    # filehandle.variables['A06'][:] = a06.astype('f4')
    # filehandle.variables['A09'][:] = a09.astype('f4')

    filehandle.close()

def compose_filename(data):
    timestamp = data.timestamp
    timestamp_string = timestamp.strftime('%Y%m%d_%H%M')
    filename = 'avhrr_iceconc_{}_arctic.nc'.format(timestamp_string)
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

    # p.add_argument("-s", "--satellite_name")
    # p.add_argument('-b', '--channels', nargs='+',
    #                      help='Input channels',
    #                      type=str)
    args = p.parse_args()
    areas_filepath = args.areas_file[0]

    ''' Test wic-classifier in python.'''

    # Read in test coefficients file for daytime
    # coeffs_filename = 'coeffPDF_daytime_mean-std-line_v2p1.txt'
    coeffs_filename = args.coeffs[0] # "./coeffPDF_daytime_mean-std-line_v2p2-misha.txt"
    coeffs = read_coeffs_from_file(coeffs_filename)
    sensor_name = args.sensor[0]

    # reduce coefficients to just the ones needed for this sensor
    coeffs = coeffs[np.logical_and(coeffs['sensor']==sensor_name, coeffs['datatype']=='gac')]

    # Read in test AVHRR swath file, with lat/lon info (for trimming)
    avhrr_filepath = args.input_file[0]
    avhrr_dirpath = os.path.dirname(avhrr_filepath)
    avhrr_basename = os.path.basename(avhrr_filepath)
    avhrr = AvhrrData(avhrr_filepath, locations=True)

    angles_basename = avhrr_basename.replace('avhrr', 'sunsatangles')
    angles_filepath = os.path.join(avhrr_dirpath, angles_basename)
    angles = AngleData(angles_filepath)

    pigobs, pcgobs, pwgobs = calc_wic_prob_day_twi(coeffs, avhrr, angles)

    sic = compute_sic(avhrr.data[1], pigobs, pwgobs, pcgobs, avhrr.lon, avhrr.lat)

    swath_def = pr.geometry.SwathDefinition(lons=avhrr.lon, lats=avhrr.lat)
    area_def = pr.utils.load_area(areas_filepath, 'nsidc_stere_north_10k')

    valid_input_index, valid_output_index, index_array, distance_array = \
                            kd_tree.get_neighbour_info(swath_def,
                                                           area_def, 10000,
                                                       neighbours=1)

    sic_res = kd_tree.get_sample_from_neighbour_info('nn', area_def.shape, sic,
                                              valid_input_index, valid_output_index,
                                              index_array, fill_value=-32767)

    pice_res = kd_tree.get_sample_from_neighbour_info('nn', area_def.shape, pigobs,
                                              valid_input_index, valid_output_index,
                                              index_array, fill_value=-32767)

    pwater_res = kd_tree.get_sample_from_neighbour_info('nn', area_def.shape, pwgobs,
                                              valid_input_index, valid_output_index,
                                              index_array, fill_value=-32767)

    pclouds_res = kd_tree.get_sample_from_neighbour_info('nn', area_def.shape, pcgobs,
                                              valid_input_index, valid_output_index,
                                              index_array, fill_value=-32767)

    a06_res = kd_tree.get_sample_from_neighbour_info('nn', area_def.shape, avhrr.data[1],
                                              valid_input_index, valid_output_index,
                                              index_array, fill_value=-32767)

    a09_res = kd_tree.get_sample_from_neighbour_info('nn', area_def.shape, avhrr.data[2],
                                              valid_input_index, valid_output_index,
                                              index_array, fill_value=-32767)

    sic_filename = compose_filename(avhrr)
    output_path = os.path.join(args.output_dir[0], sic_filename)

    # Load OSI SAF landmask and apply to resampled SIC array
    land_mask_filepath = os.path.join(os.path.dirname(
                                      os.path.abspath(__file__)),
		                      'resources',
                                      'land_mask.npz')

    land_mask = get_osisaf_land_mask(land_mask_filepath)
    # import ipdb; ipdb.set_trace()
    sic_res = apply_mask(land_mask, sic_res)

    (area_def.lons, area_def.lats) = area_def.get_lonlats()
    save_sic(output_path,
                 sic_res,
                 pice_res,
                 pwater_res,
                 pclouds_res,
                 a06_res,
                 a09_res,
                 (avhrr.timestamp - datetime.datetime(1970,1,1)).total_seconds(),
                 area_def.lons,
                 area_def.lats)

    soz = angles.data[1]
    day_mask = soz <= 90

    area = "nsidc_stere_north_10k"


def calc_wic_prob_day_twi(coeffs, avhrr, angles):
    ''' Calculate water-ice-cloud daytime and twilight probabilities.'''

    # Use A06 or not
    useA06 = True

    # Defining undef values
    iceclflundef = -1.0
    prob_undef   = -999.0

    # Put data in variables with shorter name, just for simplicity
    A06 = avhrr.data[1]
    A09 = avhrr.data[2]
    A16 = avhrr.data[6]
    T37 = avhrr.data[3]
    T11 = avhrr.data[4]
    SOZ = angles.data[1]

    # Turn the SOZ numbers into ints suitable for indexing (truncate float to int)
    SOZ = SOZ.astype(np.int16)
    coeff_indices = np.where(SOZ <= 90, SOZ, 0)
    #day_mask = SOZ <= 90
    #SOZ = ma.masked_greater(SOZ, 90)

    # Decide which data to use.
    # Especially important for chosing between re1.6/re0.6 and bt3.7-bt11. Prefer to use 1.6 if available.
    useA0906  = (A06 >= 0.00001) * (A09 >= 0.00001)
    useA0906 *= (A06 <= 100.0) * (A09 <= 100.0)
    useA0906 *= (SOZ > 0.0) * (SOZ < 90.0)
    useA16    = (A16 > 0.00001) * (A16 <= 100.0)
    useA16   *= (SOZ > 0.0) * (SOZ < 90.0)
    useT37    = np.invert(useA16) * (T37 > 50.0)
    useT37   *= (T37 < 400.0) * (T11 > 50.0)
    useT37   *= (T11 < 400.0) * (SOZ > 0.0) * (SOZ < 90.0)    
    # useA16  = np.array([False]) #$  = (A16 > 0.00001) * (A16 <= 100.0)
    # useT37  = np.array([False]) #  = np.invert(useA16) * (T37 > 50.0)

    # Combine the input variables to the features to be used
    A0906 = (A09 / A06)
    try:
        A1606 = (A16 / A06) 
    except:
        pass
   
    T3711 = (T37-T11)
    
    # Estimate the probability of getting the observed A09/A06 and A16/A06 or T37-T11
    # given ice, cloud or water.

    # First, always calculate the re0.9/re0.6 and re0.6 probabilities and put in VAR1 and VAR2 variables
    var = 're09/re06'
    cloud_mean, cloud_std, sea_mean, sea_std, ice_mean, ice_std = get_coeffs_for_var(coeffs, var, 
                                                                indices=coeff_indices)

    pVAR1gc = normalpdf(A0906, cloud_mean, cloud_std)
    pVAR1gw = normalpdf(A0906, sea_mean, sea_std)
    pVAR1gi = normalpdf(A0906, ice_mean, ice_std)

    if (useA06):
        var = 're06'
        cloud_mean, cloud_std, sea_mean, sea_std, ice_mean, ice_std = get_coeffs_for_var(coeffs, var, 
                                                                                         indices=coeff_indices)
        pVAR2gc = normalpdf(A06, cloud_mean, cloud_std)
        pVAR2gw = normalpdf(A06, sea_mean, sea_std)
        pVAR2gi = normalpdf(A06, ice_mean, ice_std)
    

    # Calculate the re1.6/re0.6 probabilities if any of the input data have the 1.6um channel
    if (useA16.any()):
        var = 're16/re06'
        cloud_mean, cloud_std, sea_mean, sea_std, ice_mean, ice_std = get_coeffs_for_var(coeffs, var,
                                                                        indices=coeff_indices)
        
        pA1606gc = normalpdf(A1606, cloud_mean, cloud_std)
        pA1606gw = normalpdf(A1606, sea_mean, sea_std)
        pA1606gi = normalpdf(A1606, ice_mean, ice_std)

    # Calculate the bt3.7-bt11 probabilities if any of the input data have the 3.7um channel
    if (useT37.any()):
        var = 'bt37-bt11'
        cloud_mean, cloud_std, sea_mean, sea_std, ice_mean, ice_std = get_coeffs_for_var(coeffs, var,
                                                                            indices=coeff_indices)

        pT3711gc = normalpdf(T3711, ice_mean, ice_std)
        pT3711gw = normalpdf(T3711, cloud_mean, cloud_std)
        pT3711gi = normalpdf(T3711, sea_mean, sea_std)

    # Put the re1.6/re0.6 based or bt3.7-bt11 based probabilites in VAR2 variables
    # re1.6/re0.6 have first priority. First fill with bt3.7-bt11, then overwrite
    # with re1.6/re0.6
    anyVAR3 = False
    if (useT37.any()):
        anyVAR3 = True
        pVAR3gw = pT3711gw.copy()
        pVAR3gi = pT3711gi.copy()
        pVAR3gc = pT3711gc.copy()
        if (useA16.any()):
            # TODO: check... only replacing values where there is an A16 value
            pVAR3gw[useA16] = pA1606gw[useA16]
            pVAR3gi[useA16] = pA1606gi[useA16]
            pVAR3gc[useA16] = pA1606gc[useA16]
    elif (useA16.any()):
        anyVAR3 = True
        pVAR3gw = pA1606gw.copy()
        pVAR3gi = pA1606gi.copy()
        pVAR3gc = pA1606gc.copy()

    useVAR3  = useA16 + useT37

    
    # Use Bayes theorem and estimate probability for ice, water and cloud.
    # Assumes equal apriori probability for ice, water, and cloud.

    # First, calculate only using VAR1 and VAR2
    if (useA06) :
        psumVAR12 = (pVAR1gi*pVAR2gi) + (pVAR1gw*pVAR2gw) + (pVAR1gc*pVAR2gc)
        pigobs = ((pVAR1gi*pVAR2gi) / psumVAR12)
        pwgobs = ((pVAR1gw*pVAR2gw) / psumVAR12)
        pcgobs = ((pVAR1gc*pVAR2gc) / psumVAR12)
    else:
        psumVAR1 = pVAR1gi + pVAR1gw + pVAR1gc
        pigobs  = (pVAR1gi/psumVAR1)
        pwgobs  = (pVAR1gw/psumVAR1)
        pcgobs  = (pVAR1gc/psumVAR1)

    # Then, calculate using both VAR1, VAR2 and VAR3, and overwrite the results from
    # using only VAR1 where there are valid VAR2 data.
    if (anyVAR3):
        if (useA06):
            psumVAR123 = (pVAR1gi*pVAR2gi*pVAR3gi) + (pVAR1gw*pVAR2gw*pVAR3gw) + (pVAR1gc*pVAR2gc*pVAR3gc)
            pigobs[useVAR3] = ((pVAR1gi[useVAR3]*pVAR2gi[useVAR3]*pVAR3gi[useVAR3]) / psumVAR123[useVAR3])
            pwgobs[useVAR3] = ((pVAR1gw[useVAR3]*pVAR2gw[useVAR3]*pVAR3gw[useVAR3]) / psumVAR123[useVAR3])
            pcgobs[useVAR3] = ((pVAR1gc[useVAR3]*pVAR2gc[useVAR3]*pVAR3gc[useVAR3]) / psumVAR123[useVAR3])
        else:
            psumVAR123 = (pVAR1gi*pVAR3gi) + (pVAR1gw*pVAR3gw) + (pVAR1gc*pVAR3gc)
            pigobs[useVAR3] = ((pVAR1gi[useVAR3]*pVAR3gi[useVAR3]) / psumVAR123[useVAR3])
            pwgobs[useVAR3] = ((pVAR1gw[useVAR3]*pVAR3gw[useVAR3]) / psumVAR123[useVAR3])
            pcgobs[useVAR3] = ((pVAR1gc[useVAR3]*pVAR3gc[useVAR3]) / psumVAR123[useVAR3])
            


    # Quality check on the probabilities
    falsevalue = (pwgobs > 1.0)*(pwgobs < 0.0)
    falsevalue *= (pigobs > 1.0)*(pigobs < 0.0)
    falsevalue *= (pcgobs > 1.0)*(pcgobs < 0.0)
    falsevalue *= ma.is_masked(A06)
    
    pigobs[falsevalue] = prob_undef
    pcgobs[falsevalue] = prob_undef
    pwgobs[falsevalue] = prob_undef

    # Could also return simple flag.
    #iceclflag = (pigobs*100.0 + pcgobs*100.0)
    #iceclflag[falsevalue] = iceclflundef
    #iceclflag[iceclflag > 100] = iceclflundef

    return (pigobs,pcgobs,pwgobs)

def get_coeffs_for_var(coeffs, var, indices=None):
    ''' Return mean and std arrays for given variable.'''
    a = coeffs[coeffs['var']==var]['coeffs']
    cloud_coeffs, sea_coeffs, ice_coeffs, snow_coeffs, land_coeffs= coeffs[coeffs['var']==var]['coeffs']

    # Get array of coefficients indexed by integer SOZ, split into mean and std 
    # TODO: this is gross
    cloud_mean, cloud_std = cloud_coeffs[indices][:,:,1], cloud_coeffs[indices][:,:,2]
    sea_mean, sea_std = sea_coeffs[indices][:,:,1], sea_coeffs[indices][:,:,2],
    ice_mean, ice_std = ice_coeffs[indices][:,:,1], ice_coeffs[indices][:,:,2]

    return cloud_mean, cloud_std, sea_mean, sea_std, ice_mean, ice_std

def read_coeffs_from_file(filename):
    ''' Read coeffs from file and return numpy rec array with all values.

    Values of array:
        'var': variable
        'sensor': avhrr.11
        'wic': [water|ice|cloud]
        'coeffs': (91,2) array of (mean, std)
     '''
    dtype = [('sensor', 'a15'),
             ('datatype','a6'),
             ('SOT','a9'),           
            ('var', 'a18'), 
            ('wic', 'a8'),
             ('FCD', 'a8'), 
            ('coeffs', 'f4', (91,3))
            #('coeffs', 
            #   [('mean', 'f4'), 
            #   ('std', 'f4')], 91)
            ]
    coeffs = np.genfromtxt(filename, dtype=dtype)

    return coeffs

def normalpdf(x, mu, sigma):
    ''' Calculate Gaussian distribution.

        mu: mean of distribution, 
        sigma: std of distribution, 
        x: value for which to calculate
    '''

    gpdf = np.zeros(len(x))
    gpdf = (1.0/(((2.0*np.pi)**0.5)*sigma)) * np.exp(-1.0*((x-mu)**2.0) / (2.0* (sigma**2.0)))
    return(gpdf)

def lognormalpdf(x, mu, sigma):
    ''' Calculate log-normal distribution.

    mu: mean of log of distribution, 
    sigma: std of log of distribution, 
    x: value for which to calculate
    '''

    gpdf = np.zeros(len(x))
    z = np.power((np.log(x)-mu),2.0)/(sigma**2.0)
    e = math.e**(-0.5*z**2.0)
    C = x*sigma*math.sqrt(2.0*math.pi)
    gpdf = 1.0*e/C
    return(gpdf)


# AVHRR code by Cristina Luis

# TODO: easily named access for each channel
# TODO: check out pypps to share ideas
# TODO: return appropriate type based on filename (one base class?)

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
            image = 'image' + str(ch)
            self.data[ch] = np.float32(avhrr[image]['data'].value)
            offset = avhrr[image]['what'].attrs['offset'] # To check for K?
            gain = avhrr[image]['what'].attrs['gain']
            self.nodata[ch] = avhrr[image]['what'].attrs['nodata']
            self.missingdata[ch] = avhrr[image]['what'].attrs['missingdata']

            mask = (self.data[ch] != self.missingdata[ch]) & (self.data[ch] != self.nodata[ch])
            self.data[ch][mask] = self.data[ch][mask] * gain + offset

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

if __name__ == '__main__':
    main()
