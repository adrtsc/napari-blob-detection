# -*- coding: utf-8 -*-
"""
Created on Thu May 13 06:13:09 2021

@author: Adria
"""
%gui qt5
import napari_blob_detection
from skimage import data
import napari
from skimage.color import rgb2gray

image = data.hubble_deep_field()[0:500, 0:500]
image_gray = rgb2gray(image)

viewer = napari.Viewer()
viewer.add_image(image_gray, name='img')

# In the viewer window navigate to:
# "Plugins" -> "Add Dock Widget" -> "napari-blob-detection" -> "detect_blobs"

viewer.window.add_dock_widget(blob_detection())

filter_widget = filter_widget()
viewer.window.add_dock_widget(filter_widget)
