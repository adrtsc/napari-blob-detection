import numpy as np
import collections
from napari_blob_detection.measure_blobs import measure_blobs
from napari_blob_detection.utils import diam_to_napari
from skimage.feature import blob_log, blob_dog
from trackpy import locate
from napari.types import LayerDataTuple
from napari.layers import Image
from magicgui import magic_factory


@magic_factory(layer={'tooltip': ('2D or 3D ndarray. Input grayscale image, '
                                  'blobs are assumed to be light on dark '
                                  'background (white on black).')},
               min_sigma={'widget_type': 'LiteralEvalLineEdit',
                          'tooltip': ('scalar or sequence of scalars, '
                                      'optional. σ \u2248 diameter/(2*√2). '
                                      'The minimum standard deviation for '
                                      'Gaussian kernel. Keep this low to '
                                      'detect smaller blobs. The standard '
                                      'deviations of the Gaussian filter are '
                                      'given for each axis as a sequence, or '
                                      'as a single number, in which case it '
                                      'is equal for all axes.')},
               max_sigma={'widget_type': 'LiteralEvalLineEdit',
                          'tooltip': ('scalar or sequence of scalars, '
                                      'optional. The maximum standard '
                                      'deviation for Gaussian kernel. Keep '
                                      'this high to detect larger blobs. The '
                                      'standard deviations of the Gaussian '
                                      'filter are given for each axis as a '
                                      'sequence, or as a single number, in '
                                      'which case it is equal for all axes.')},
               num_sigma={
                   'tooltip': ('int, optional.The number of intermediate '
                               'values of standard deviations to consider '
                               'between min_sigma and max_sigma.')},
               overlap={
                   'tooltip': ('float, optional. A value between 0 and 1. If '
                               'the area of two blobs overlaps by a fraction '
                               'greater than threshold, the smaller blob is '
                               'eliminated.')},
               log_scale={
                   'tooltip': ('bool, optional. If set intermediate values '
                               'of standard deviations are interpolated '
                               'using a logarithmic scale to the base 10. '
                               'If not, linear interpolation is used.')},
               exclude_border={
                   'tooltip': ('tuple of ints, int, or False, optional. '
                               'If tuple of ints, the length of the tuple '
                               'must match the input array’s dimensionality. '
                               'Each element of the tuple will exclude peaks '
                               'from within exclude_border-pixels of the '
                               'border of the image along that dimension. If '
                               'nonzero int, exclude_border excludes peaks '
                               'from within exclude_border-pixels of the '
                               'border of the image. If zero or False, '
                               'peaks are identified regardless of their '
                               'distance from the border.')},
               call_button="Detect blobs",
               marker={"choices": ['disc', 'ring', 'diamond'],
                       'tooltip': 'marker to represent the detected blobs'},
               threshold={"step": 10e-15,
                          "tooltip": ('float, optional. The absolute lower '
                                      'bound for scale space maxima. '
                                      'Local maxima smaller than thresh are '
                                      'ignored. Reduce this to detect blobs '
                                      'with less intensities.')})
def laplacian_of_gaussian(
        layer: Image,
        min_sigma=3,
        max_sigma=3,
        num_sigma=1,
        threshold=0.003000,
        overlap=0.50,
        log_scale=False,
        exclude_border=False,
        marker='disc') -> LayerDataTuple:
    """Detect blobs in image, return points layer with spots"""

    # adjust sigmas in case only one scalar is entered:

    if isinstance(min_sigma, collections.abc.Sequence) == False:
        min_sigma = np.repeat(min_sigma, 3)
    if isinstance(max_sigma, collections.abc.Sequence) == False:
        max_sigma = np.repeat(max_sigma, 3)

    # get arguments of function
    kwargs = locals()
    kwargs = {el: val for (el, val) in kwargs.items() if
              el not in ['layer', 'marker']}

    blobs = measure_blobs(layer.data,
                          blob_log,
                          measure_features=False,
                          **kwargs)

    output = blobs[['timepoint',
                    'centroid-0',
                    'centroid-1',
                    'centroid-2']]

    sizes = diam_to_napari(blobs[['size-time',
                                  'size-0',
                                  'size-1',
                                  'size-2']])

    return (output, {'size': sizes,
                     'features': blobs,
                     'symbol': marker,
                     'edge_color': 'white',
                     'face_color': 'transparent',
                     'opacity': 0.5,
                     'scale': layer.scale},
            'points')


@magic_factory(layer={'tooltip': ('2D or 3D ndarray. Input grayscale image, '
                                  'blobs are assumed to be light on dark '
                                  'background (white on black).')},
               min_sigma={'widget_type': 'LiteralEvalLineEdit',
                          'tooltip': ('scalar or sequence of scalars, '
                                      'optional. σ \u2248 diameter/(2*√2). '
                                      'The minimum standard deviation for '
                                      'Gaussian kernel. Keep this low to '
                                      'detect smaller blobs. The standard '
                                      'deviations of the Gaussian filter are '
                                      'given for each axis as a sequence, or '
                                      'as a single number, in which case it '
                                      'is equal for all axes.')},
               max_sigma={'widget_type': 'LiteralEvalLineEdit',
                          'tooltip': ('scalar or sequence of scalars, '
                                      'optional. The maximum standard '
                                      'deviation for Gaussian kernel. Keep '
                                      'this high to detect larger blobs. The '
                                      'standard deviations of the Gaussian '
                                      'filter are given for each axis as a '
                                      'sequence, or as a single number, in '
                                      'which case it is equal for all axes.')},
               overlap={
                   'tooltip': ('float, optional. A value between 0 and 1. If '
                               'the area of two blobs overlaps by a fraction '
                               'greater than threshold, the smaller blob is '
                               'eliminated.')},
               exclude_border={
                   'tooltip': ('tuple of ints, int, or False, optional. '
                               'If tuple of ints, the length of the tuple '
                               'must match the input array’s dimensionality. '
                               'Each element of the tuple will exclude peaks '
                               'from within exclude_border-pixels of the '
                               'border of the image along that dimension. If '
                               'nonzero int, exclude_border excludes peaks '
                               'from within exclude_border-pixels of the '
                               'border of the image. If zero or False, '
                               'peaks are identified regardless of their '
                               'distance from the border.')},
               call_button="Detect blobs",
               marker={"choices": ['disc', 'ring', 'diamond'],
                       'tooltip': 'marker to represent the detected blobs'},
               threshold={"step": 10e-15,
                          "tooltip": ('float, optional. The absolute lower '
                                      'bound for scale space maxima. '
                                      'Local maxima smaller than thresh are '
                                      'ignored. Reduce this to detect blobs '
                                      'with less intensities.')})
def difference_of_gaussian(
        layer: Image,
        min_sigma=3,
        max_sigma=3,
        sigma_ratio=1.6,
        threshold=0.003000,
        overlap=0.50,
        exclude_border=False,
        marker='disc') -> LayerDataTuple:
    """Detect blobs in image, return points layer with spots"""

    # adjust sigmas in case only one scalar is entered:

    if isinstance(min_sigma, collections.abc.Sequence) == False:
        min_sigma = np.repeat(min_sigma, 3)
    if isinstance(max_sigma, collections.abc.Sequence) == False:
        max_sigma = np.repeat(max_sigma, 3)

    # get arguments of function
    kwargs = locals()
    kwargs = {el: val for (el, val) in kwargs.items() if
              el not in ['layer', 'marker']}

    blobs = measure_blobs(layer.data,
                          blob_dog,
                          measure_features=False,
                          **kwargs)

    output = blobs[['timepoint',
                    'centroid-0',
                    'centroid-1',
                    'centroid-2']]

    sizes = diam_to_napari(blobs[['size-time',
                                  'size-0',
                                  'size-1',
                                  'size-2']])

    return (output, {'size': sizes,
                     'features': blobs,
                     'symbol': marker,
                     'edge_color': 'white',
                     'face_color': 'transparent',
                     'opacity': 0.5,
                     'scale': layer.scale},
            'points')


@magic_factory(layer={'tooltip': ('2D or 3D ndarray. Input grayscale image, '
                                  'blobs are assumed to be light on dark '
                                  'background (white on black).')},
               call_button="Detect blobs",
               marker={"choices": ['disc', 'ring', 'diamond'],
                       'tooltip': 'marker to represent the detected blobs'},
               threshold={"step": 10e-15,
                          "tooltip": ('float, optional. The absolute lower '
                                      'bound for scale space maxima. '
                                      'Local maxima smaller than thresh are '
                                      'ignored. Reduce this to detect blobs '
                                      'with less intensities.')})
def tp_locate(
        layer: Image,
        diameter=3,
        minmass=None,
        maxsize=None,
        separation=None,
        noise_size=1,
        smoothing_size=None,
        threshold=None,
        invert=False,
        percentile=64,
        topn=None,
        preprocess=True,
        max_iterations=10,
        filter_before=None,
        filter_after=None,
        characterize=True,
        engine='auto',
        marker='disc') -> LayerDataTuple:
    """Detect blobs in image, return points layer with spots"""

    # get arguments of function
    kwargs = locals()
    kwargs = {el: val for (el, val) in kwargs.items() if
              el not in ['layer', 'marker']}

    blobs = []

    for idx, i in enumerate(list(img)):
        current_blobs = locate(i, **kwargs)
        current_blobs['timepoint'] = idx
        current_blobs['size-time'] = 1

        blobs.append(current_blobs)

    blobs = pd.concat(blobs)

    blobs.rename(columns={'y': 'centroid-1',
                          'x': 'centroid-2',
                          'size': 'size-1'}, inplace=True)
    blobs['size-2'] = blobs['size-1']
    blobs['size-2'] = blobs['size-2']
    blobs['size-0'] = 0

    # adjust size to represent diameter rather radius

    blobs['size-0'] = blobs['size-0'] * 2
    blobs['size-1'] = blobs['size-1'] * 2
    blobs['size-2'] = blobs['size-2'] * 2

    output = blobs[['timepoint',
                    'centroid-0',
                    'centroid-1',
                    'centroid-2']]

    sizes = diam_to_napari(blobs[['size-time',
                                  'size-0',
                                  'size-1',
                                  'size-2']])

    return (output, {'size': sizes,
                     'features': blobs,
                     'symbol': marker,
                     'edge_color': 'white',
                     'face_color': 'transparent',
                     'opacity': 0.5,
                     'scale': layer.scale},
            'points')