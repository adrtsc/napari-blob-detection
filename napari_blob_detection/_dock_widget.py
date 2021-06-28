"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
import numpy as np
import pandas as pd
import pickle


from napari_plugin_engine import napari_hook_implementation
from skimage.feature import blob_dog, blob_log, blob_doh
from napari.types import LayerDataTuple, ImageData
from napari.layers import Image, Points
from magicgui import magic_factory
from napari import Viewer
from pathlib import Path
from enum import Enum
from math import sqrt
from .svm import SVM
from .measure_blobs import measure_blobs


def init(widget):
    
    # this option should only be visible for blob_dog
    widget.sigma_ratio.visible = False
    
    # update available options for each detector
    @widget.detector.changed.connect
    def update_options(event):
        
        if widget.detector.value.value == "blob_log":
            widget.num_sigma.visible = True
            widget.log_scale.visible = True
            widget.sigma_ratio.visible = False
            
        elif widget.detector.value.value == "blob_dog":
            widget.num_sigma.visible = False
            widget.log_scale.visible = False
            widget.sigma_ratio.visible = True
        
            
        elif widget.detector.value.value == "blob_doh":
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

@magic_factory(layer={'tooltip': '2D or 3D ndarray. Input grayscale image, blobs are assumed to be light on dark background (white on black).'},
               detector={'tooltip': 'Detection algorithm to use. LoG = Laplacian of Gaussian, DoG = Difference of Gaussian, DoH = Determinant of Hessian'},
               min_sigma={'tooltip': 'scalar or sequence of scalars, optional. σ \u2248 diameter/(2*√2). The minimum standard deviation for Gaussian kernel. Keep this low to detect smaller blobs. The standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number, in which case it is equal for all axes.'},
               max_sigma={'tooltip': 'scalar or sequence of scalars, optional. The maximum standard deviation for Gaussian kernel. Keep this high to detect larger blobs. The standard deviations of the Gaussian filter are given for each axis as a sequence, or as a single number, in which case it is equal for all axes.'},
               num_sigma={'tooltip': 'int, optional.The number of intermediate values of standard deviations to consider between min_sigma and max_sigma.'},
               overlap={'tooltip': 'float, optional. A value between 0 and 1. If the area of two blobs overlaps by a fraction greater than threshold, the smaller blob is eliminated.'},
               log_scale={'tooltip': 'bool, optional. If set intermediate values of standard deviations are interpolated using a logarithmic scale to the base 10. If not, linear interpolation is used.'},
               exclude_border={'tooltip': 'tuple of ints, int, or False, optional. If tuple of ints, the length of the tuple must match the input array’s dimensionality. Each element of the tuple will exclude peaks from within exclude_border-pixels of the border of the image along that dimension. If nonzero int, exclude_border excludes peaks from within exclude_border-pixels of the border of the image. If zero or False, peaks are identified regardless of their distance from the border.'},
               call_button="Detect blobs",
               marker={"choices": ['disc', 'ring', 'diamond'], 
                       'tooltip': 'marker to represent the detected blobs'},
               threshold={"step": 10e-15,
                          "tooltip": 'float, optional. The absolute lower bound for scale space maxima. Local maxima smaller than thresh are ignored. Reduce this to detect blobs with less intensities.'},
               widget_init=init)
               
def blob_detection(
    layer: Image,
    viewer: Viewer,
    detector: Detector,
    min_sigma=1.00,
    max_sigma=50.00,
    sigma_ratio=1.6,
    num_sigma=10,
    threshold=0.200000,
    overlap=0.50,
    log_scale=False,
    exclude_border=False,
    marker='disc') -> LayerDataTuple:
    """Detect blobs in image, returny points layer with spots"""
    
    if detector.value == "blob_log":
        result = blob_log(layer.data,
                          min_sigma=min_sigma,
                          max_sigma=max_sigma,
                          num_sigma=num_sigma,
                          threshold=threshold,
                          overlap=overlap,
                          log_scale=log_scale,
                          exclude_border=exclude_border)
    
    elif detector.value == "blob_dog":
        result =  blob_dog(layer.data, min_sigma=min_sigma,
                           max_sigma=max_sigma,
                           sigma_ratio=sigma_ratio,
                           threshold=threshold,
                           overlap=overlap,
                           exclude_border=exclude_border)
    
    elif detector.value == "blob_doh":
        result =  blob_doh(layer.data, min_sigma=min_sigma,
                           max_sigma=max_sigma,
                           num_sigma=num_sigma,
                           threshold=threshold,
                           overlap=overlap,
                           log_scale=log_scale)
        
    # turn results into pd DataFrame
    
    # add empty z-dimension if not present
    
    if result.shape[1] == 3:
        result = np.pad(result, [(0, 0), (1, 0)])
    
    result = pd.DataFrame(result, columns=['z_coordinates',
                                           'y_coordinates',
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
                      'name': detector.value, 'opacity': 0.5}, 'points')


def filter_init(widget):
    
    widget.call_button.visible = False
    
    @widget.initialize_filter.changed.connect
    def measure(event):
        
        coordinates = widget.points_layer.value.data
        sizes = widget.points_layer.value.size
                  
        data_df = measure_blobs(coordinates, sizes, widget.img.value.data)
        widget.data_df.value = data_df
        widget.filter_df.value = pd.DataFrame()
        
        widget.result_path.visible = True
        widget.save_results.visible = True
        widget.add_filter.visible = True
        
        
        print("Filter initialized.")
        
    widget.subfilter_counter=0
    @widget.add_filter.changed.connect
    def add_filter(event):
        
        if isinstance(widget.data_df.value, pd.DataFrame):
    
            sf = subfilter()
            sf.label = " " 
            sf.name = "subfilter_%02d" % widget.subfilter_counter
            
            widget.insert(-4, sf)
    
            widget.subfilter_counter += 1
            
            # update min/max threshold
            
            sf.threshold.max = np.max(
                  widget.data_df.value[sf.feature.value.value])
            sf.threshold.min = np.min(
                widget.data_df.value[sf.feature.value.value])
            
            
            @sf.threshold.changed.connect
            def apply_filter(event):
                widget.filter_df.value[sf.name] =  (widget.data_df.value[
                    sf.feature.value.value] >= sf.threshold.value)
                
                widget.filter_df.value.all(axis=1)
                
                data_df = widget.data_df.value
                df_filtered = data_df.loc[widget.filter_df.value.all(axis=1)]

                if widget.points_layer.value.ndim == 3:
                    output = df_filtered[['z_coordinates',
                                          'y_coordinates',
                                          'x_coordinates']]
                    new_size = df_filtered[['size_z', 'size_y', 'size_x']]

                elif widget.points_layer.value.ndim == 2:
                    output = df_filtered[['y_coordinates',
                                          'x_coordinates']]
                    new_size = df_filtered[['size_y', 'size_x']]

                widget.points_layer.value.data = output
                widget.points_layer.value.size = new_size
                widget.points_layer.value.selected_data.clear()
                widget.points_layer.value.refresh()
                
            @sf.feature.changed.connect
            def update_threshold(event):
                
                # update min/max threshold
            
                sf.threshold.max = np.max(
                      widget.data_df.value[sf.feature.value.value])
                sf.threshold.min = np.min(
                    widget.data_df.value[sf.feature.value.value])
                sf.threshold.value = np.min(
                widget.data_df.value[sf.feature.value.value])
                
            @sf.delete.changed.connect
            def delete_subfilter(event):
                sf.threshold.value = sf.threshold.min
                widget.remove(sf.name)
                
                
        else:
            raise Exception("Filter was not initialized.") 
            
    @widget.save_results.changed.connect
    def save_results(event):
        
        if len(widget.filter_df.value > 0):
            result_df = widget.data_df.value.loc[
                widget.filter_df.value.all(axis=1)]
        else:
            result_df = widget.data_df.value
            
        result_df.to_csv(widget.result_path.value)
        
class Feature(Enum):
    """A set of valid arithmetic operations for image_arithmetic.

    To create nice dropdown menus with magicgui, it's best
    (but not required) to use Enums.  Here we make an Enum
    class for all of the image math operations we want to
    allow.
    """
    z_coordinates = "z_coordinates"
    y_coordinates = "y_coordinates"
    x_coordinates = "x_coordinates"
    min_intensity = "min_intensity"
    max_intensity = "max_intensity"
    mean_intensity = "mean_intensity"
    var_intensity = "var_intensity"
    mean_background_intensity = "mean_background_intensity"
    SNR = "SNR"
    size_z = "size_z"
    size_y = "size_y"
    size_x = "size_x"


@magic_factory(add_filter={"widget_type": "PushButton", 'visible': False},
               layout='vertical',
               result_path={'mode': 'w', 'label': 'save results',
                            'visible': False},
               save_results={'widget_type': 'PushButton', 'visible': False},
               initialize_filter={'widget_type': 'PushButton'},
               widget_init=filter_init)
def filter_widget(img: Image,
                  points_layer: Points,
                  data_df=Image,
                  filter_df=Image,
                  add_filter=0,
                  initialize_filter=0,
                  result_path=Path(),
                  save_results=0):
    pass


@magic_factory(auto_call=True,
               threshold={'label': " ", "widget_type": "FloatSlider"},
               delete={"widget_type": "PushButton"},
               layout='horizontal')
def subfilter(feature: Feature,
              threshold=0,
              delete=0):
    pass


##############################################################################


# class Filter(FunctionGui):
   
#     def __init__(self):
       
#         super().__init__(Filter.call,
#                          layout="vertical",
#                          param_options={"add_filter": {'widget_type': 'PushButton'}})
        
#         self.filters = 
        
#         @self.add_filter.changed.connect
#         def add_filter(event):
#             subfilter = Subfilter()
#             self.insert(-1, subfilter)
            
#             @subfilter.threshold.changed.connect
#             def test(event):
                
#                 data_df = widget.data_df.value
#                 df_filtered = data_df.loc[
#                 data_df[widget.feature.value.value] >= widget.threshold.value]
            
            
        
        
        
#     def call(img: ImageData,
#              points_layer: Points,
#              add_filter=0,
#              test=0,
#              data_df=Image,
#              filters=Image) -> LayerDataTuple:
#         print("works")
        
        
        
# class Subfilter(FunctionGui):
    
#     def __init__(self):
    
#         super().__init__(Subfilter.call,
#                          auto_call=True,
#                          layout="vertical",
#                          param_options={"threshold": {'widget_type': 'FloatSlider'}})
        
#     def call(feature: Feature,
#              threshold=0,
#              ):
#         pass
    
    
    

##############################################################################
def selector_init(widget):
    
    @widget.initialize_layers.changed.connect
    def initialize_layers(event):
        widget.points_layer.value.current_size=widget.clf_layer.value.size.mean()
        widget.points_layer.value.current_face_color = "yellow"
        widget.points_layer.value.current_edge_color = "yellow"
        widget.points_layer.value.mode = 'add'
        
        # to-do: make the points_layer the active layer upon initialization
    
    @widget.blob_class.changed.connect
    def update_blob_color(event):
        if widget.blob_class.value == 1:
            widget.points_layer.value.current_face_color = "yellow"
            widget.points_layer.value.current_edge_color = "yellow"
        elif widget.blob_class.value == 2:
            widget.points_layer.value.current_face_color = "gray"
            widget.points_layer.value.current_edge_color = "gray"
    
    @widget.save_classifier.changed.connect
    def save_classifier(event):
        
        with open(widget.clf_path.value, "wb") as fp:   #Pickling
            pickle.dump(widget.clf, fp) 
            
        print('classifer has been saved')
        
    @widget.apply_classifier.changed.connect
    def apply_classifier(event):
        blobs = measure_blobs(widget.clf_layer.value.data,
                              widget.clf_layer.value.size,
                              widget.clf_img_layer.value.data)
        
        widget.result = widget.clf.classify(blobs)
        
        pos = widget.result.loc[widget.result['classification'] == 1]
        
        widget.viewer.value.add_points(pos[['z_coordinates',
                                            'y_coordinates',
                                            'x_coordinates']],
                                       size=pos[['size_z',
                                                 'size_y',
                                                 'size_x']],
                                       name='result')
        
     
@magic_factory(blob_class={'widget_type': 'RadioButtons',
                             "orientation": "horizontal",
                             'value': 1,
                             "choices": [("foreground", 1), ("background", 2)]},
               initialize_layers={'widget_type': 'PushButton'},
               save_classifier={'widget_type': 'PushButton'},
               clf_path={'mode': 'w', 'label': 'save classifier'},
               apply_classifier={'widget_type': 'PushButton'},
               widget_init=selector_init)
def selection_widget(points_layer: Points,
                     img_layer: Image,
                     clf_layer : Points,
                     clf_img_layer: Image,
                     viewer: Viewer,
                     initialize_layers=0,
                     blob_class=2,
                     clf_path = Path(),
                     save_classifier=0,
                     apply_classifier=0):
    
    data_df = measure_blobs(points_layer.data,
                            points_layer.size,
                            img_layer.data)
    labels = np.unique(np.mean(points_layer.face_color, axis=1),
                       return_inverse=True)[1]
    
    weights = np.unique(labels, return_counts=True)[0]
    selection_widget.training_data = data_df.drop(['z_coordinates',
                                                   'y_coordinates',
                                                   'x_coordinates'], axis=1)
    
    selection_widget.clf = SVM(training_data=selection_widget.training_data,
                               labels=labels,
                               split_ratio=0.2,
                               weights={0:1, 1:weights[0]/weights[1]},
                               columns=selection_widget.training_data.columns)
    
    selection_widget.clf.train_svm()
    
    print("classifier has been trained")
    
    
    ''''this part isn't really needed, just use "add points" tool 
    that is already part of the layer
    '''
    
    # @points_layer.mouse_drag_callbacks.append
    # def add_points(layer, event):
    #     points_layer.add(points_layer.coordinates)
    #     yield
    #     while event.type == 'mouse_move':
    #         points_layer.add(points_layer.coordinates)
    #         yield
    

    
    
def loader_init(widget):
    
    @widget.blob_path.changed.connect
    def update_default_choice(event):
        
        widget.df = pd.read_csv(widget.blob_path.value)
        
        widget.z_coordinates.choices = list(widget.df.columns)
        widget.x_coordinates.choices = list(widget.df.columns)
        widget.y_coordinates.choices = list(widget.df.columns)
        
        if "z_coordinates" in widget.df.columns:
            widget.z_coordinates.value = "z_coordinates"
        if "x_coordinates" in widget.df.columns:
            widget.y_coordinates.value = "y_coordinates"
        if "y_coordinates" in widget.df.columns:
            widget.x_coordinates.value = "x_coordinates"
    
     
@magic_factory(blob_path={'mode': 'r', 'label': 'path to csv'},
               z_coordinates={"choices": [""]},
               y_coordinates={"choices": [""]},
               x_coordinates={"choices": [""]},
               widget_init=loader_init)
def loading_widget(blob_path=Path(),
                   z_coordinates="",
                   y_coordinates="",
                   x_coordinates="",) -> LayerDataTuple:
    
    output = loading_widget.df[['z_coordinates',
                                'y_coordinates',
                                'x_coordinates']]
    size = loading_widget.df[['size_z',
                              'size_y',
                              'size_x']]
    
    return (output, {'size': size, 'symbol': 'disc',
                      'name':'blobs', 'opacity': 0.5}, 'points')

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [blob_detection, filter_widget, selection_widget, loading_widget]



