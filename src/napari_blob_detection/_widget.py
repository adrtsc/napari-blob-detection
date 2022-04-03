"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""
import numpy as np
import pandas as pd
import trackpy as tp
import pickle
from napari.types import LayerDataTuple
from napari_blob_detection.measure_blobs import measure_coordinates
from napari_blob_detection.svm import SVM
from napari_blob_detection.utils import diam_from_napari, diam_to_napari
from napari_blob_detection.detectors import (laplacian_of_gaussian,
                                             difference_of_gaussian,
                                             tp_locate)
from enum import Enum
from napari.layers import Image, Points
from magicgui import magic_factory
from napari import Viewer
from pathlib import Path

DETECTOR_MAPPING = {"blob_log": laplacian_of_gaussian,
                    "blob_dog": difference_of_gaussian,
                    "trackpy_locate": tp_locate}

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
    trackpy_locate = "trackpy_locate"

    # not using difference of hessian here, because it only works on 2D images


def blob_detection_init(widget):
    widget.call_button.visible = False
    detector_widget = DETECTOR_MAPPING[widget.detector.value.value]
    detector_widget = detector_widget()
    detector_widget.label = ""
    detector_widget.name = 'detector_widget'
    widget.insert(1, detector_widget)
    widget.detector_widget.parent_changed()

    @widget.detector.changed.connect
    def update(event):
        if hasattr(widget, 'detector_widget'):
            widget.remove('detector_widget')
        detector_widget = DETECTOR_MAPPING[widget.detector.value.value]
        detector_widget = detector_widget()
        detector_widget.label = ""
        detector_widget.name = 'detector_widget'
        widget.insert(1, detector_widget)
        widget.detector_widget.parent_changed()

@magic_factory(widget_init=blob_detection_init)
def blob_detection(detector: Detector):

    pass


def selector_init(widget):
    @widget.initialize_layers.changed.connect
    def initialize_layers(event):
        widget.annotation_layer.value.current_size = widget.points_layer.value.size[
                                                     :, 1:].mean()
        widget.annotation_layer.value.current_face_color = "yellow"
        widget.annotation_layer.value.current_edge_color = "yellow"
        widget.annotation_layer.value.opacity = 0.5
        widget.annotation_layer.value.mode = 'add'

        if hasattr(widget, "data_df") == False:
            widget.data_df = pd.DataFrame(columns=['timepoint',
                                                   'centroid-0',
                                                   'centroid-1',
                                                   'centroid-2',
                                                   'size-time',
                                                   'size-0',
                                                   'size-1',
                                                   'size-2'])

        # to-do: make the annotation_layer the active layer upon initialization

        @widget.blob_class.changed.connect
        def update_blob_color(event):
            if widget.blob_class.value == 1:
                widget.annotation_layer.value.current_face_color = "yellow"
                widget.annotation_layer.value.current_edge_color = "yellow"
            elif widget.blob_class.value == 2:
                widget.annotation_layer.value.current_face_color = "gray"
                widget.annotation_layer.value.current_edge_color = "gray"

    @widget.save_classifier.changed.connect
    def save_classifier(event):

        with open(widget.clf_path.value, "wb") as fp:  # Pickling
            pickle.dump(widget.clf, fp)

        print('classifer has been saved')

    @widget.add_training_data.changed.connect
    def add_training_data(event):
        coordinates = widget.annotation_layer.value.data
        sizes = diam_from_napari(widget.annotation_layer.value.size)
        data_df = measure_coordinates(coordinates,
                                      sizes,
                                      widget.img_layer.value.data)
        labels = \
            np.unique(
                np.mean(widget.annotation_layer.value.face_color, axis=1),
                return_inverse=True)[1]

        widget.data_df = data_df
        widget.lbls = labels

        print("training data was added")

    @widget.apply_classifier.changed.connect
    def apply_classifier(event):
        coordinates = widget.points_layer.value.data
        sizes = diam_from_napari(widget.points_layer.value.size)
        blobs = measure_coordinates(coordinates,
                                    sizes,
                                    widget.clf_img_layer.value.data)

        widget.result = widget.clf.classify(blobs)

        pos = widget.result.loc[widget.result['classification'] == 1]

        coordinates = pos[['timepoint',
                           'centroid-0',
                           'centroid-1',
                           'centroid-2']]

        sizes = diam_to_napari(pos[['size-time',
                                    'size-0',
                                    'size-1',
                                    'size-2']])

        widget.viewer.value.add_points(coordinates,
                                       size=sizes,
                                       name='result',
                                       edge_color='yellow',
                                       face_color='transparent',
                                       opacity=0.5,
                                       scale=widget.points_layer.value.scale)


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
def selection_widget(annotation_layer: Points,
                     img_layer: Image,
                     points_layer: Points,
                     clf_img_layer: Image,
                     viewer: Viewer,
                     initialize_layers=0,
                     blob_class=2,
                     clf_path=Path(),
                     save_classifier=0,
                     apply_classifier=0,
                     add_training_data=0):
    weights = np.unique(selection_widget.lbls, return_counts=True)[0]
    selection_widget.training_data = selection_widget.data_df.drop(
        ['timepoint',
         'centroid-0',
         'centroid-1',
         'centroid-2',
         'size-time'], axis=1)

    # drop columns that are constant across training data:
    selection_widget.training_data = selection_widget.training_data.loc[
                                     :, (selection_widget.training_data !=
                                         selection_widget.training_data.iloc[
                                             0]).any()]

    selection_widget.clf = SVM(training_data=selection_widget.training_data,
                               labels=selection_widget.lbls,
                               split_ratio=0.2,
                               weights={0: 1, 1: weights[0] / weights[1]},
                               columns=selection_widget.training_data.columns)

    selection_widget.clf.train_svm()

    print("classifier has been trained")


def filter_init(widget):
    widget.call_button.visible = False

    @widget.initialize_filter.changed.connect
    def measure(event):

        coordinates = widget.points_layer.value.data
        sizes = diam_from_napari(widget.points_layer.value.size)

        data_df = measure_coordinates(coordinates, sizes,
                                      widget.img.value.data)
        data_df[['size-time', 'size-0', 'size-1', 'size-2']] = diam_to_napari(
            data_df[['size-time', 'size-0', 'size-1', 'size-2']])
        widget.data_df.value = data_df
        widget.filter_df.value = pd.DataFrame()

        widget.result_path.visible = True
        widget.save_results.visible = True
        widget.add_filter.visible = True

        print("Filter initialized.")

    widget.subfilter_counter = 0

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
                widget.data_df.value[sf.feature.value.value]) * 1.05
            sf.threshold.min = np.min(
                widget.data_df.value[sf.feature.value.value]) * 0.95

            @sf.threshold.changed.connect
            def apply_filter(event):

                if sf.min_max.value == 1:
                    widget.filter_df.value[sf.name] = (widget.data_df.value[
                                                           sf.feature.value.value] >= sf.threshold.value)
                if sf.min_max.value == 2:
                    widget.filter_df.value[sf.name] = (widget.data_df.value[
                                                           sf.feature.value.value] <= sf.threshold.value)

                widget.filter_df.value.all(axis=1)

                data_df = widget.data_df.value
                df_filtered = data_df.loc[widget.filter_df.value.all(axis=1)]

                output = df_filtered[['timepoint',
                                      'centroid-0',
                                      'centroid-1',
                                      'centroid-2']]
                new_size = df_filtered[['size-time',
                                        'size-0',
                                        'size-1',
                                        'size-2']]

                widget.points_layer.value.data = output
                widget.points_layer.value.size = new_size
                widget.points_layer.value.selected_data.clear()
                widget.points_layer.value.refresh()

            @sf.min_max.changed.connect
            def update_min_max(event):
                if sf.min_max.value == 1:
                    sf.threshold.value = sf.threshold.min
                if sf.min_max.value == 2:
                    sf.threshold.value = sf.threshold.max

            @sf.feature.changed.connect
            def update_threshold(event):

                # update min/max threshold

                sf.threshold.max = np.max(
                    widget.data_df.value[sf.feature.value.value]) * 1.05
                sf.threshold.min = np.min(
                    widget.data_df.value[sf.feature.value.value]) * 0.95

                if sf.min_max.value == 1:
                    sf.threshold.value = np.min(
                        widget.data_df.value[sf.feature.value.value])
                if sf.min_max.value == 2:
                    sf.threshold.value = np.max(
                        widget.data_df.value[sf.feature.value.value])

            @sf.delete.changed.connect
            def delete_subfilter(event):

                if sf.min_max.value == 1:
                    sf.threshold.value = sf.threshold.min
                if sf.min_max.value == 2:
                    sf.threshold.value = sf.threshold.max

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
    z_coordinates = "centroid-0"
    y_coordinates = "centroid-1"
    x_coordinates = "centroid-2"
    min_intensity = "min_intensity"
    max_intensity = "max_intensity"
    mean_intensity = "mean_intensity"
    var_intensity = "var_intensity"
    mean_background_intensity = "mean_background_intensity"
    SNR = "SNR"
    size_z = 'size-0'
    size_y = 'size-1'
    size_x = 'size-2'


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
               min_max={'label': " ", "widget_type": "RadioButton",
                        "choices": [("min", 1), ("max", 2)],
                        'tooltip': "min: filter out values lower than threshold."
                                   " max: filter out values higher than threshold"},
               threshold={'label': " ", "widget_type": "FloatSlider",
               'backend_kwargs':{'width': 100}},
               delete={"widget_type": "PushButton"},
               layout='horizontal')
def subfilter(feature: Feature,
              min_max=1,
              threshold=0,
              delete=0):
    pass


@magic_factory(call_button="track blobs",)
def tracking_widget(points_layer: Points,
                    search_range=5,
                    memory=0) -> LayerDataTuple:

    df = pd.DataFrame(points_layer.data,
                      columns=['frame', 'z', 'y', 'x'])

    tracks = tp.link(df, search_range=search_range, memory=memory)
    tracks = tracks[['particle', 'frame', 'z', 'y', 'x']]

    return (tracks, {'scale': points_layer.scale},
            'tracks')
