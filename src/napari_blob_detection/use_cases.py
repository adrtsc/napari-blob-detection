import napari
from skimage import data
import numpy as np
import pandas as pd
from skimage.filters import gaussian
from skimage.draw import disk
from skimage.util import random_noise

# 2D case:
viewer = napari.Viewer()
img = data.binary_blobs(length=100,
                        n_dim=2,
                        blob_size_fraction=0.01,
                        volume_fraction=0.001)

img = gaussian(img, 1)

img = random_noise(img)

img = np.expand_dims(img, axis=(0, 1))

viewer.add_image(img, name="img")

blobs = measure_blobs(img)

viewer.add_points(blobs[['timepoint', 'centroid-0', 'centroid-1', 'centroid-2']])

# 2Dt case:

img = [data.binary_blobs(length=100,
                   n_dim=2,
                   blob_size_fraction=0.01,
                   volume_fraction=0.001) for i in range(0, 10)]

img = [gaussian(i, 1) for i in img]

img = np.stack(img)

img = random_noise(img)

img = np.expand_dims(img, axis=1)


viewer = napari.Viewer()
viewer.add_image(img, name="img")

blobs = measure_blobs(img)

viewer.add_points(blobs[['timepoint', 'centroid-0', 'centroid-1', 'centroid-2']])

# 3D case:

img = data.binary_blobs(length=100,
                        n_dim=3,
                        blob_size_fraction=0.01,
                        volume_fraction=0.001)

img = gaussian(img, 1)

img = np.expand_dims(img, axis=0)

viewer.add_image(img, name="img")

blobs = measure_blobs(img)

viewer.add_points(blobs[['timepoint', 'centroid-0', 'centroid-1', 'centroid-2']])


# 3Dt case:

img = [data.binary_blobs(length=100,
                   n_dim=3,
                   blob_size_fraction=0.01,
                   volume_fraction=0.001) for i in range(0, 10)]

img = [gaussian(i, 1) for i in img]

img = np.stack(img)

viewer = napari.Viewer()
viewer.add_image(img, name="img")

blobs = measure_blobs(img)

viewer.add_points(blobs[['timepoint', 'centroid-0', 'centroid-1', 'centroid-2']])



def measure_blobs(img):

    blobs = []

    for idx, i in enumerate(list(img)):
        current_blobs = blob_log(i,
                                 threshold=0.003,
                                 num_sigma=1,
                                 min_sigma=3,
                                 max_sigma=3)

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


test = blob_detection()
viewer.window.add_dock_widget(test)

test2 = selection_widget()
viewer.window.add_dock_widget(test2)

measure_coordinates(test2.points_layer.value.data,
                    test2.points_layer.value.size,
                    test2.clf_img_layer.value.data)


import h5py
import yaml
from pathlib import Path

settings_path = Path(r"Y:\PhD\Code\Python\lca\scripts\settings\20211111_settings.yml")
with open(settings_path, 'r') as stream:
    settings = yaml.safe_load(stream)

channel = 'sdcRFP590-JF549'
site = 1
hdf5_path = Path(settings['paths']['hdf5_path'])

n_tp = 5

with h5py.File(hdf5_path.joinpath('site_%04d.hdf5' % site), "a") as file:
    intensity_image = file[f'intensity_images/{channel}'][0:n_tp, :, :, :]

viewer = napari.Viewer()
viewer.add_image(intensity_image)