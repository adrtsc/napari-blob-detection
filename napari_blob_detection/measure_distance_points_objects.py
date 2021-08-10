import pandas as pd
import numpy as np
from scipy.spatial import KDTree

def measure_distance_points_objects(blobs, mask, anisotropy):

    df = blobs
    df['z_coordinates'] = df['z_coordinates']*anisotropy

    seg_coords = pd.DataFrame()
    seg_coords['z'], seg_coords['y'], seg_coords['x'] = np.where(mask == 1)
    seg_coords['z'] = seg_coords['z']*anisotropy

    tree = KDTree(seg_coords)
    dist, points = tree.query(df[['z_coordinates', 'y_coordinates', 'x_coordinates']], 1)
    blobs['distance_object'] = dist

    # same for distance to background (or how deep in a speckle they are)
    bg_coords = pd.DataFrame()
    bg_coords['z'], bg_coords['y'], bg_coords['x'] = np.where(mask == 0)

    tree_bg = KDTree(bg_coords)
    dist, points = tree_bg.query(df[['z_coordinates', 'y_coordinates', 'x_coordinates']], 1)
    blobs['distance_bg'] = dist

    return blobs
