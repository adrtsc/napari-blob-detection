import numpy as np
import random
import pandas as pd

def random_points_mask(mask, n=500):
    mask_pixels = np.where(mask>0)
    rnd_centers = random.sample(range(1, len(mask_pixels[0])), n)

    random_blobs = pd.DataFrame()

    random_blobs['z_coordinates'] = mask_pixels[0][rnd_centers]
    random_blobs['y_coordinates'] = mask_pixels[1][rnd_centers]
    random_blobs['x_coordinates'] = mask_pixels[2][rnd_centers]

    return random_blobs