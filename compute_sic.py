""" Compute single swath Sea Ice Concentration

Use numpy arrays -- the output of the wic_classify_test.py

"""
import os
import numpy as np
import numpy.ma as ma
from matplotlib import pyplot as plt
from scipy import stats


def load_data(*data_arrays_path):
    data_list = []
    for i in data_arrays_path:
        if os.path.exists(i):
            data_list.append(np.load(i))
        else:
            raise OSError('No path: {}'.format(i))
    return data_list


def plot_swath(data, lons, lats):
    """ Plot swath data

    Using Arctic data only
    """
    import pyresample as pr

    plt.figure(figsize=(20,10))

    swath_def = pr.geometry.SwathDefinition(lons=lons, lats=lats)
    area_def = pr.utils.load_area('./areas.cfg', 'istjenesten_main_4k')
    result = pr.kd_tree.resample_nearest(swath_def,
                                         data,
                                         area_def,
                                         radius_of_influence=10000,
                                         fill_value=None,
                                         nprocs=4)
    pr.plot.save_quicklook('out.png',
                            area_def,
                            result,
                            vmin=0,
                            coast_res = 'l',
                            vmax=100)


def compute_sic():
    """compute sic

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
    ( data, pice, pwater, pclouds, lons, lats ) = load_data('a09.npy',
                                             'pice.npy',
                                             'pwater.npy',
                                             'pclouds.npy',
                                             'lons.npy',
                                             'lats.npy')

    ice_mask = pice > 0.99
    water_mask = pwater > 0.99
    lats_mask = lats > 65

    ice_data = ma.array(data, mask = (ice_mask * lats_mask) == false)
    ice_hist = np.histogram(ice_data[ice_data.mask == false], 20)
    ice_max = np.mean(ice_hist[1])

    water_data = ma.array(data, mask = (water_mask * lats_mask) == false)
    water_hist = np.histogram(water_data[water_data.mask == false], 20)
    water_max = np.min(water_hist[1])

    print ice_max, water_max

    only_ice_mask = (pice > pwater) * (pice > pclouds) * (lats > 65)
    only_ice_data = ma.array(data, mask = only_ice_mask == false)

    # compute regression coefficients
    slope, intercept, r_value, p_value, std_err = stats.linregress((water_max, 20), (0, 100))
    sic = slope * only_ice_data + intercept

    plot_swath(sic, lons, lats)
    # plt.imshow(sic, vmin = 0, vmax=100); plt.colorbar()
    # plt.show()

def main():
    compute_sic()

if __name__ == "__main__":
    main()
