from napari_blob_detection import (blob_detection,
                                   filter_widget,
                                   selection_widget,
                                   tracking_widget)
import numpy as np
from skimage import data
from skimage.color import rgb2gray

def test_blob_docking(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.window.add_dock_widget(blob_detection())
    assert "Dock widget 1" in viewer.window._dock_widgets,\
        'blob detection has not been docked'

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

def test_tracker_docking(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.window.add_dock_widget(tracking_widget())
    assert "Dock widget 1" in viewer.window._dock_widgets,\
        'tracking widget has not been docked'

def test_blob_detection(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = blob_detection()
    viewer.window.add_dock_widget(widget)

    image = data.hubble_deep_field()[0:500, 0:500]
    image_gray = rgb2gray(image)

    image_gray = np.expand_dims(image_gray, (0, 1))

    viewer.add_image(image_gray)

    widget.detector_widget.__call__()

    assert 'Points' in viewer.layers, 'blob detection fails'

def test_filter_widget(make_napari_viewer):
    # make napari viewer
    viewer = make_napari_viewer()

    # add sample image
    image = data.hubble_deep_field()[0:500, 0:500]
    image_gray = rgb2gray(image)
    image_gray = np.expand_dims(image_gray, (0, 1))

    viewer.add_image(image_gray)

    # add sample blobs
    coords = np.random.randint(0, 500, [10, 2])
    coords = np.c_[np.zeros(np.shape(coords)), coords]

    viewer.add_points(coords)
    len_data = len(viewer.layers['coords'].data)

    # add widget
    widget = filter_widget()
    viewer.window.add_dock_widget(widget)

    # initialize the widget
    widget.initialize_filter.clicked.__call__('initialize_filter')

    # add a filter
    widget.add_filter.clicked.__call__('add_filter')

    # filter out all the blobs
    widget.subfilter_00.feature.value = widget.subfilter_00.feature.choices[1]
    widget.subfilter_00.threshold.value = widget.subfilter_00.threshold.max

    len_data_filtered = len(viewer.layers['coords'].data)

    assert len_data_filtered < len_data, 'blob filtering fails'

    # remove filter
    widget.subfilter_00.delete.clicked.__call__('delete')