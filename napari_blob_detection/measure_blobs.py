import numpy as np
import pandas as pd
from skimage.draw import disk

def measure_blobs(coordinates, sizes, img):
    if coordinates.shape[1] == 2:
        coordinates = np.pad(coordinates, [(0, 0), (1, 0)])
        sizes = np.pad(sizes, [(0, 0), (1, 0)])

    data = np.column_stack((coordinates, sizes))
    data_df = pd.DataFrame(data, columns=['z_coordinates',
                                          'y_coordinates',
                                          'x_coordinates',
                                          'size_z',
                                          'size_y',
                                          'size_x'])

    min_intensity = []
    max_intensity = []
    mean_intensity = []
    var_intensity = []
    mean_bg_intensity = []

    for index, row in data_df.iterrows():

        if img.ndim == 2:

            c_img = img

        else:

            c_img = img[row['z_coordinates'].astype("int"), :, :]

        rr, cc = disk(tuple(row[1:3]), row['size_y'],
                      shape=np.shape(c_img))

        rr_bg, cc_bg = disk(tuple(row[1:3]),
                            2 * row['size_y'],
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

    ''' Tried to have measurements for spheres for each spot
    but I think that was computationally too expensive'''

    # for index, row in data_df.iterrows():

    #     sim = nltools.Simulator()
    #     sim.brain_mask = sim.to_nifti(np.ones(widget.img.value.shape, dtype="int"))
    #     sphere = sim.sphere(p=row[0:3], r=row['size_y'])
    #     bg_sphere = sim.sphere(p=row[0:3], r=row['size_y']*2)

    #     pixels = widget.img.value[sphere==1]

    #     pixels_bg = widget.img.value[bg_sphere==1]

    #     n_pixels = len(pixels)
    #     n_pixels_bg = len(pixels_bg)

    #     mean_bg_intensity.append((np.sum(pixels_bg) - np.sum(pixels))
    #                               / (n_pixels_bg - n_pixels))

    #     mean_intensity.append(np.mean(pixels))

    #     min_intensity.append(np.min(pixels))
    #     max_intensity.append(np.max(pixels))
    #     var_intensity.append(np.var(pixels))

    data_df['min_intensity'] = min_intensity
    data_df['max_intensity'] = max_intensity
    data_df['mean_intensity'] = mean_intensity
    data_df['var_intensity'] = var_intensity
    data_df['mean_background_intensity'] = mean_bg_intensity
    data_df['SNR'] = np.array(mean_intensity) / np.array(mean_bg_intensity)

    return data_df