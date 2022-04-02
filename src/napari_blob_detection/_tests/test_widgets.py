from napari_blob_detection import (blob_detection,
                                   filter_widget,
                                   selection_widget)
import numpy as np
from skimage import data
from skimage.color import rgb2gray

def test_mock():
    pass


def test_blob_docking(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.window.add_dock_widget(blob_detection())
    assert "Dock widget 1" in viewer.window._dock_widgets,\
        'blob detection has not been docked'
'''

def test_filter_docking(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.window.add_dock_widget(filter_widget())
    assert "Dock widget 1" in viewer.window._dock_widgets,\
        'filter widget has not been docked'

def test_selection_docking(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.window.add_dock_widget(selection_widget())
    assert "Dock widget 1" in viewer.window._dock_widgets,\
        'selection widget has not been docked'
'''