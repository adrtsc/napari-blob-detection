import numpy as np
import pandas as pd
from skimage.draw import disk


def measure_blobs(img, func, measure_features=False, **kwargs):

    blobs = []

    for idx, i in enumerate(list(img)):
        current_blobs = func(i, **kwargs)

        current_blobs = pd.DataFrame(current_blobs, columns=['centroid-0',
                                                             'centroid-1',
                                                             'centroid-2',
                                                             'size-0',
                                                             'size-1',
                                                             'size-2'])

        current_blobs['timepoint'] = idx
        current_blobs['size-time'] = 1

        blobs.append(current_blobs)

    blobs = pd.concat(blobs)

    # adjust size to represent diameter rather than sigma

    blobs['size-0'] = blobs['size-0'] * np.sqrt(2) * 2
    blobs['size-1'] = blobs['size-1'] * np.sqrt(2) * 2
    blobs['size-2'] = blobs['size-2'] * np.sqrt(2) * 2

    if measure_features:
        coordinates = blobs[['timepoint',
                             'centroid-0',
                             'centroid-1',
                             'centroid-2']]

        sizes = blobs[['size-time', 'size-0', 'size-1', 'size-2']]

        blobs = measure_coordinates(coordinates, sizes, img)

    return blobs


def measure_coordinates(coordinates, diameters, img):

    data = np.column_stack((coordinates, diameters))
    blobs = pd.DataFrame(data, columns=['timepoint',
                                        'centroid-0',
                                        'centroid-1',
                                        'centroid-2',
                                        'size-time',
                                        'size-0',
                                        'size-1',
                                        'size-2'])

    # define measurements

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
        # assuming scale in x,y are the same for the moment
        rr, cc = disk(tuple(row[['centroid-1', 'centroid-2']]),
                      row['size-1']/2,
                      shape=np.shape(c_img))

        rr_bg, cc_bg = disk(tuple(row[['centroid-1', 'centroid-2']]),
                            row['size-1'],
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

    return blobs
