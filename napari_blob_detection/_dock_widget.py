"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton, QComboBox, QLabel
from magicgui import magic_factory
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from enum import Enum
from napari.types import LayerDataTuple, ImageData
from napari import Viewer
import numpy as np
import pandas as pd
from skimage.draw import disk


def init(widget):
    
    # this option should only be visible for blob_dog
    widget.sigma_ratio.visible = False
    
    # update available options for each detector
    @widget.approach.changed.connect
    def update_options(event):
        
        if widget.approach.value.value == "blob_log":
            widget.num_sigma.visible = True
            widget.log_scale.visible = True
            widget.sigma_ratio.visible = False
            
        elif widget.approach.value.value == "blob_dog":
            widget.num_sigma.visible = False
            widget.log_scale.visible = False
            widget.sigma_ratio.visible = True
        
            
        elif widget.approach.value.value == "blob_doh":
            widget.num_sigma.visible = True
            widget.log_scale.visible = True
            widget.sigma_ratio.visible = False
            widget.exclude_border.visible = False
            


class Detector(Enum):
    """A set of valid arithmetic operations for image_arithmetic.

    To create nice dropdown menus with magicgui, it's best
    (but not required) to use Enums.  Here we make an Enum
    class for all of the image math operations we want to
    allow.
    """
    # dropdown options for detectors
    LoG = "blob_log"
    DoG = "blob_dog"
    DoH = "blob_doh"

@magic_factory(call_button="Detect blobs",
               marker={"choices": ['disc', 'ring', 'diamond']},
               widget_init=init)
def detect_blobs(
    layer: "napari.layers.Image",
    viewer: Viewer,
    approach: Detector,
    min_sigma=1,
    max_sigma=50,
    sigma_ratio=1.6,
    num_sigma=10,
    threshold=0.2,
    overlap=0.5,
    log_scale=False,
    exclude_border=False,
    marker = 'disc') -> LayerDataTuple:
    """Detect blobs in image, returny points layer with spots"""
    
    if approach.value == "blob_log":
        result = blob_log(layer.data,
                          min_sigma=min_sigma,
                          max_sigma=max_sigma,
                          num_sigma=num_sigma,
                          threshold=threshold,
                          overlap=overlap,
                          log_scale=log_scale,
                          exclude_border=exclude_border)
    
    elif approach.value == "blob_dog":
        result =  blob_dog(layer.data, min_sigma=min_sigma,
                           max_sigma=max_sigma,
                           sigma_ratio=sigma_ratio,
                           threshold=threshold,
                           overlap=overlap,
                           exclude_border=exclude_border)
    
    elif approach.value == "blob_doh":
        result =  blob_dog(layer.data, min_sigma=min_sigma,
                           max_sigma=max_sigma,
                           num_sigma=num_sigma,
                           threshold=threshold,
                           overlap=overlap,
                           log_scale=log_scale)
        
    # turn results into pd DataFrame
    
    result = pd.DataFrame(result, columns=['y_coordinates',
                                           'x_coordinates',
                                           'size'])
        
    # get radius
    result.iloc[:, -1] = result.iloc[:, -1] * sqrt(2)

    # get spot size
    size = result.iloc[:, -1]*2
    output = result.iloc[:, 0:-1].astype(int)
    
    filter_widget = filter_blobs(
        blobs_df = {'value': result},
        l_name = {'value': approach.value},
        layer = {'value': layer.data},
        marker = {'value': marker})
    
    if "Filter widget" in viewer.window._dock_widgets:
        viewer.window.remove_dock_widget(viewer.window._dock_widgets["Filter widget"])
        
    viewer.window.add_dock_widget(filter_widget, name="Filter widget")

    return (output, {'size': size, 'symbol': marker,
                     'name':approach.value, 'opacity': 0.5}, 'points')

def init_filter(widget):
    
    widget.threshold.max = np.max(
          widget.blobs_df.value[widget.feature.value.value])
    widget.threshold.min = np.min(
        widget.blobs_df.value[widget.feature.value.value])
    
    @widget.feature.changed.connect
    def update_max_threshold(event):
        widget.threshold.max = np.max(
            widget.blobs_df.value[widget.feature.value.value])
        widget.threshold.min = np.min(
            widget.blobs_df.value[widget.feature.value.value])
        widget.threshold.value = np.min(
            widget.blobs_df.value[widget.feature.value.value])
        

    mean_intensity = []
    for blob in np.array(widget.blobs_df.value):
        rr, cc = disk(tuple(blob[0:2]), blob[-1], shape=np.shape(widget.layer.value))
        pixels = widget.layer.value[rr, cc]
        mean_intensity.append(np.mean(pixels))
    widget.blobs_df.value['mean_intensity'] = mean_intensity 


class Filter(Enum):
    """A set of valid arithmetic operations for image_arithmetic.

    To create nice dropdown menus with magicgui, it's best
    (but not required) to use Enums.  Here we make an Enum
    class for all of the image math operations we want to
    allow.
    """
    y_coordinates = "y_coordinates"
    x_coordinates = "x_coordinates"
    mean_intensity = "mean_intensity"
    size = "size"


@magic_factory(auto_call=True,
               threshold={'label': " ", "widget_type": "FloatSlider"},
               layout='horizontal',
               widget_init=init_filter)
def filter_blobs(feature: Filter,
                 l_name=None,
                 layer=None,
                 blobs_df=None,
                 marker=None,
                 threshold=0) -> LayerDataTuple:

    filtered_blobs = blobs_df.loc[blobs_df[feature.value] > threshold]
    output = filtered_blobs.iloc[:, 0:2].astype(int)
    new_size = filtered_blobs['size']*2
    
    return (output, {'size': new_size, 'symbol': marker,
                     'name':l_name}, 'points')


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [detect_blobs, filter_blobs]