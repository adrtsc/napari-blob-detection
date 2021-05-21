"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
import numpy as np
import pandas as pd

from napari_plugin_engine import napari_hook_implementation
from skimage.feature import blob_dog, blob_log, blob_doh
from napari.types import LayerDataTuple, ImageData
from napari.layers import Image, Points
from magicgui import magic_factory
from skimage.draw import disk
from napari import Viewer
from enum import Enum
from math import sqrt


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
def blob_detection(
    layer: Image,
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
        result =  blob_doh(layer.data, min_sigma=min_sigma,
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
    
    # create filter widget and dock to viewer
    filter_instance = filter_widget()
    viewer.window.add_dock_widget(filter_instance, name="Filter widget") 

    return (output, {'size': size, 'symbol': marker,
                      'name':approach.value, 'opacity': 0.5}, 'points')
    
    
    
def filter_init(widget):
    
    @widget.feature.changed.connect
    def update_max_threshold(event):

        widget.threshold.max = np.max(
            widget.data_df.value[widget.feature.value.value])
        widget.threshold.min = np.min(
            widget.data_df.value[widget.feature.value.value])
        widget.threshold.value = np.min(
            widget.data_df.value[widget.feature.value.value])
    
    @widget.measure.changed.connect
    def measure(event):
    
        data = np.column_stack((widget.points_layer.value.data,
                                widget.points_layer.value.size))
        
        data_df = pd.DataFrame(data, columns=['y_coordinates', 
                                              'x_coordinates',
                                              'size_y',
                                              'size_x'])
    
        mean_intensity = []
        for index, row in data_df.iterrows():
            rr, cc = disk(tuple(row[0:2]), row['size_y'],
                          shape=np.shape(widget.img.value))
            pixels = widget.img.value[rr, cc]
            mean_intensity.append(np.mean(pixels))
        
        data_df['mean_intensity'] = mean_intensity 
        
        
        widget.data_df.value = data_df
        
        # update the threshold min/max
        
        widget.threshold.max = np.max(
              widget.data_df.value[widget.feature.value.value])
        widget.threshold.min = np.min(
            widget.data_df.value[widget.feature.value.value])
        
    @widget.threshold.changed.connect
    def apply_filter(event):
    
        data_df = widget.data_df.value
        df_filtered = data_df.loc[
            data_df[widget.feature.value.value] >= widget.threshold.value]
        output = df_filtered[['y_coordinates', 'x_coordinates']]
        new_size = df_filtered[['size_y', 'size_x']]
        widget.points_layer.value.data = output
        widget.points_layer.value.size = new_size

        
class Feature(Enum):
    """A set of valid arithmetic operations for image_arithmetic.

    To create nice dropdown menus with magicgui, it's best
    (but not required) to use Enums.  Here we make an Enum
    class for all of the image math operations we want to
    allow.
    """
    y_coordinates = "y_coordinates"
    x_coordinates = "x_coordinates"
    mean_intensity = "mean_intensity"
    size_y = "size_y"
    size_x = "size_x"


@magic_factory(call_button={'label': ' ', 'visible': False},
           threshold={'label': " ", "widget_type": "FloatSlider"},
           measure={'label': 'measure', 'widget_type': 'PushButton'},
           layout='horizontal',
           widget_init=filter_init)
def filter_widget(feature: Feature,
                  img: ImageData,
                  points_layer: Points,
                  threshold=0,
                  measure=0,
                  data_df=Image) -> LayerDataTuple:
    pass
    

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [blob_detection, filter_widget]
