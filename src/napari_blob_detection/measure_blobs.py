import numpy as np
import pandas as pd
from skimage.draw import disk
from skimage.feature import blob_log


def measure_blobs(img,
                  min_sigma=3,
                  max_sigma=3,
                  num_sigma=1,
                  threshold=0.003000,
                  overlap=0.50,
                  log_scale=False,
                  exclude_border=False,
                  ):

    blobs = []

    for idx, i in enumerate(list(img)):
        current_blobs = blob_log(i,
                                 min_sigma=min_sigma,
                                 max_sigma=max_sigma,
                                 num_sigma=num_sigma,
                                 threshold=threshold,
                                 overlap=overlap,
                                 log_scale=log_scale,
                                 exclude_border=exclude_border)

        current_blobs = pd.DataFrame(current_blobs, columns=['centroid-0',
                                                             'centroid-1',
                                                             'centroid-2',
                                                             'size'])

        current_blobs['timepoint'] = idx

        blobs.append(current_blobs)

    blobs = pd.concat(blobs)

    # additional measurements

    min_intensity = []
    max_intensity = []
    mean_intensity = []
    var_intensity = []
    mean_bg_intensity = []

    for index, row in blobs.iterrows():

        c_img = img[row['timepoint'].astype('int'),
                row['centroid-0'].astype('int'),
                :,
                :]

        rr, cc = disk(tuple(row[['centroid-1', 'centroid-2']]), row['size'],
                      shape=np.shape(c_img))

        rr_bg, cc_bg = disk(tuple(row[['centroid-1', 'centroid-2']]),
                            2 * row['size'],
                            shape=np.shape(c_img))

        pixels = c_img[rr, cc]
        pixels_bg = c_img[rr_bg, cc_bg]

        n_pixels = len(pixels)
        n_pixels_bg = len(pixels_bg)

        mean_bg_intensity.append((np.sum(pixels_bg) - np.sum(pixels))
                                 / (n_pixels_bg - n_pixels))

        mean_intensity.append(np.mean(pixels))

        min_intensity.append(np.min(pixels))
        max_intensity.append(np.max(pixels))
        var_intensity.append(np.var(pixels))


    blobs['min_intensity'] = min_intensity
    blobs['max_intensity'] = max_intensity
    blobs['mean_intensity'] = mean_intensity
    blobs['var_intensity'] = var_intensity
    blobs['mean_background_intensity'] = mean_bg_intensity
    blobs['SNR'] = np.array(mean_intensity) / np.array(mean_bg_intensity)

    # adjust size to represent diameter rather than sigma

    blobs['size'] = blobs['size']*np.sqrt(2)*2

    return blobs


def measure_coordinates(coordinates, sizes, img):

    data = np.column_stack((coordinates, sizes[:, 0]))
    blobs = pd.DataFrame(data, columns=['timepoint',
                                        'centroid-0',
                                        'centroid-1',
                                        'centroid-2',
                                        'size'])

    # additional measurements

    min_intensity = []
    max_intensity = []
    mean_intensity = []
    var_intensity = []
    mean_bg_intensity = []

    for index, row in blobs.iterrows():

        c_img = img[row['timepoint'].astype('int'),
                row['centroid-0'].astype('int'),
                :,
                :]

        rr, cc = disk(tuple(row[['centroid-1', 'centroid-2']]), row['size'],
                      shape=np.shape(c_img))

        rr_bg, cc_bg = disk(tuple(row[['centroid-1', 'centroid-2']]),
                            2 * row['size'],
                            shape=np.shape(c_img))

        pixels = c_img[rr, cc]
        pixels_bg = c_img[rr_bg, cc_bg]

        n_pixels = len(pixels)
        n_pixels_bg = len(pixels_bg)

        mean_bg_intensity.append((np.sum(pixels_bg) - np.sum(pixels))
                                 / (n_pixels_bg - n_pixels))

        mean_intensity.append(np.mean(pixels))

        min_intensity.append(np.min(pixels))
        max_intensity.append(np.max(pixels))
        var_intensity.append(np.var(pixels))


    blobs['min_intensity'] = min_intensity
    blobs['max_intensity'] = max_intensity
    blobs['mean_intensity'] = mean_intensity
    blobs['var_intensity'] = var_intensity
    blobs['mean_background_intensity'] = mean_bg_intensity
    blobs['SNR'] = np.array(mean_intensity) / np.array(mean_bg_intensity)

    # adjust size to represent diameter rather than sigma

    #blobs['size'] = blobs['size']*np.sqrt(2)*2

    return blobs