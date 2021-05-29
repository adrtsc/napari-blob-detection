# -*- coding: utf-8 -*-
"""
Created on Tue May 25 21:31:48 2021

@author: Adria
"""
from napari_blob_detection._dock_widget import blob_detection, filter_widget
import napari
from skimage import data
from skimage.color import rgb2gray


def test_blob_docking():
    
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(blob_detection)
    assert "Dock widget 1" in viewer.window._dock_widgets, 'blob detection has not been docked'


def test_filter_docking():
    
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(filter_widget())
    assert "Dock widget 1" in viewer.window._dock_widgets, 'filter widget has not been docked'


def test_blob_detection():
    
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(blob_detection)
    
    image = data.hubble_deep_field()[0:500, 0:500]
    image_gray = rgb2gray(image)
    
    viewer.add_image(image_gray)