"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""
import numpy as np
import pandas as pd
import pickle
from napari_blob_detection.measure_blobs import measure_blobs
from napari_blob_detection.measure_blobs import measure_coordinates
from napari_blob_detection.svm import SVM
from napari.types import LayerDataTuple
from napari.layers import Image, Points
from magicgui import magic_factory
from napari import Viewer
from pathlib import Path


@magic_factory(layer={'tooltip': '2D or 3D ndarray. Input grayscale image, blobs are assumed to be light on dark background (white on black).'},
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
                          "tooltip": 'float, optional. The absolute lower bound for scale space maxima. Local maxima smaller than thresh are ignored. Reduce this to detect blobs with less intensities.'})
def blob_detection(
        layer: Image,
        viewer: Viewer,
        min_sigma=3,
        max_sigma=3,
        num_sigma=1,
        threshold=0.003000,
        overlap=0.50,
        log_scale=False,
        exclude_border=False,
        marker='disc') -> LayerDataTuple:
    """Detect blobs in image, return points layer with spots"""

    # add empty z-dimension if not present

    blobs = measure_blobs(layer.data,
                          min_sigma=min_sigma,
                          max_sigma=max_sigma,
                          num_sigma=num_sigma,
                          threshold=threshold,
                          overlap=overlap,
                          log_scale=log_scale,
                          exclude_border=exclude_border
                          )
    output = blobs[['timepoint',
                    'centroid-0',
                    'centroid-1',
                    'centroid-2']]

    return (output, {'size': blobs['size'],
                     'features': blobs,
                     'symbol': marker,
                     'edge_color': 'white',
                     'face_color': 'transparent'},
            'points')


def selector_init(widget):
    @widget.initialize_layers.changed.connect
    def initialize_layers(event):
        widget.points_layer.value.current_size = widget.clf_layer.value.size.mean()
        widget.points_layer.value.current_face_color = "yellow"
        widget.points_layer.value.current_edge_color = "yellow"
        widget.points_layer.value.mode = 'add'

        if hasattr(widget, "data_df") == False:
            widget.data_df = pd.DataFrame(columns=['centroid-0',
                                                   'centroid-1',
                                                   'centroid-2',
                                                   'size'])
            widget.labels = list()

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

        with open(widget.clf_path.value, "wb") as fp:  # Pickling
            pickle.dump(widget.clf, fp)

        print('classifer has been saved')

    @widget.add_training_data.changed.connect
    def add_training_data(event):
        data_df = measure_coordinates(widget.points_layer.value.data,
                                      widget.points_layer.value.size,
                                      widget.img_layer.value.data)
        labels = \
        np.unique(np.mean(widget.points_layer.value.face_color, axis=1),
                  return_inverse=True)[1]

        widget.data_df = widget.data_df.append(data_df)
        widget.labels.extend(labels)

        print("training data was added")

    @widget.apply_classifier.changed.connect
    def apply_classifier(event):
        blobs = measure_coordinates(widget.clf_layer.value.data,
                                    widget.clf_layer.value.size,
                                    widget.clf_img_layer.value.data)

        widget.result = widget.clf.classify(blobs)

        pos = widget.result.loc[widget.result['classification'] == 1]

        widget.viewer.value.add_points(pos[['timepoint',
                                            'centroid-0',
                                            'centroid-1',
                                            'centroid-2']],
                                       size=pos[['size']],
                                       name='result',
                                       opacity=0.5,
                                       scale=widget.clf_layer.value.scale)


@magic_factory(blob_class={'widget_type': 'RadioButtons',
                           "orientation": "horizontal",
                           'value': 1,
                           "choices": [("foreground", 1), ("background", 2)]},
               initialize_layers={'widget_type': 'PushButton'},
               save_classifier={'widget_type': 'PushButton'},
               clf_path={'mode': 'w', 'label': 'save classifier'},
               apply_classifier={'widget_type': 'PushButton'},
               add_training_data={'widget_type': 'PushButton'},
               widget_init=selector_init)
def selection_widget(points_layer: Points,
                     img_layer: Image,
                     clf_layer: Points,
                     clf_img_layer: Image,
                     viewer: Viewer,
                     initialize_layers=0,
                     blob_class=2,
                     clf_path=Path(),
                     save_classifier=0,
                     apply_classifier=0,
                     add_training_data=0):
    weights = np.unique(selection_widget.labels, return_counts=True)[0]
    selection_widget.training_data = selection_widget.data_df.drop(
        ['timepoint',
         'centroid-0',
         'centroid-1',
         'centroid-2'], axis=1)

    selection_widget.clf = SVM(training_data=selection_widget.training_data,
                               labels=selection_widget.labels,
                               split_ratio=0.2,
                               weights={0: 1, 1: weights[0] / weights[1]},
                               columns=selection_widget.training_data.columns)

    selection_widget.clf.train_svm()

    print("classifier has been trained")