# napari-blob-detection

[![License](https://img.shields.io/pypi/l/napari-blob-detection.svg?color=green)](https://github.com/adrtsc/napari-blob-detection/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-blob-detection.svg?color=green)](https://pypi.org/project/napari-blob-detection)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-blob-detection.svg?color=green)](https://python.org)
[![tests](https://github.com/adrtsc/napari-blob-detection/workflows/tests/badge.svg)](https://github.com/adrtsc/napari-blob-detection/actions)
[![codecov](https://codecov.io/gh/adrtsc/napari-blob-detection/branch/main/graph/badge.svg)](https://codecov.io/gh/adrtsc/napari-blob-detection)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-blob-detection)](https://napari-hub.org/plugins/napari-blob-detection)

A napari plugin for blob detection.


<ul>
<li>blob_detection: Implements a user interface to use scikit-image blob detection algorithms (https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html).</li>

<li>filter_widget: Widget to filter the detected blobs by a blob feature value (or a combination of feature values).</li>

<li>selection_widget: Widget to manually annotate a subset of detected blobs as foreground vs. background blobs. The widget can then use the annotated blobs to train a support vector machine to classify the remaining blobs in the image.</li>

<li>tracking_widget: EXPERIMENTAL: tracking of the detected blobs in 2DT and 3DT images .</li>
</ul> 

## Installation

It's best to create a new python environment to try the plugin:

    conda create -n napari-blob-detection python=3.9
    
Activate the environment:

    conda activate napari-blob-detection
    
You will need to install napari and JupyterLab or Jupyter Notebook (if you want to test the examples)

    pip install napari[all]
    pip install jupyterlab

You can then install `napari-blob-detection` via [pip]:

    pip install git+https://github.com/adrtsc/napari-blob-detection.git

## Examples

To try the example jupyter notebooks do the following:

    git clone https://github.com/adrtsc/napari-blob-detection
    cd napari-blob-detection/examples/
    
Start JupyterLab

    jupyter lab 
    
## Using the plugin with your own data

### How to prepare your data

If you want to use the plugin with your own data, it is important that you convert your image into a 4D (t, z, y, x) array before adding it as image layer to napari. Otherwise the plugin will struggle. Often you either have to expand the dimensions of your array or rearrange the dimensions:

For example, if your image is a 2D array:

```python
import numpy as np
from skimage import io
import napari
    
# read your image    
img = io.imread('path/to/img')   
# add t and z dimensions
img = np.expand_dims(img, (0, 1))
# create napari viewer and add the image
viewer = napari.Viewer()
viewer.add_image(img)
```

In case you have a 4D array but the dimensions are not in the correct order, numpy.transpose() is helpfull to rearrange them.

```python
import numpy as np

# here is a 4D array that has the wrong order of dimensions (y, x, t, z)
img = np.empty([100, 100, 1, 10]) # img.shape is (100, 100, 1, 10)

# the following line will reorder the dimensions to (t, z, y, x)
rearranged_img = np.transpose(img, (2, 3, 0, 1)) # rearranged_img.shape is (1, 10, 100, 100)

```

### blob detection widget

To use the blob detection widget, open napari and go to "Plugins -> napari-blob-detection -> blob_detection". This will dock the blob detection widget to the napari viewer.

![grafik](https://user-images.githubusercontent.com/41194383/161416678-60d1f4f2-abce-4c0f-8c4b-64a9f5b39963.png)

You can choose which detector you would like to use in the detector drop-down. Currently, the ones available are:

<ul>
<li><a href="https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_log">LoG (laplacian of gaussian)</a> (blob_dog, skimage)</li>
<li><a href="https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.blob_dog">DoG (difference of gaussian)</a> (blob_log, skimage)</li>
<li><a href="http://soft-matter.github.io/trackpy/dev/generated/trackpy.locate.html">locate function</a> (locate, Trackpy)</li>
</ul>

The available parameters will change depending on the detector you choose. Please refer to the official implementation of the functions for details on the parameters. Once you are ready to test the parameters, click on "Detect blobs" and the plugin will create a napari points layer with the detected blobs.

### filter widget

After using the blob detection widget you can use the filter widget to filter the blobs by some feature measurements. To open the filter widget go to "Plugins -> napari-blob-detection -> filter_widget". This will dock the filter widget to the napari viewer. The filter widget will initially look like this:

![grafik](https://user-images.githubusercontent.com/41194383/161416992-e11f0cb1-eafc-40fb-b08e-f467733ecafb.png)

Make sure that you choose the right image layer and the right points layer and then click on initialize filter. The plugin will now measure some features of the blobs present in the points layer. A new button "add filter" and an interface to save the filtered blobs will appear.

![grafik](https://user-images.githubusercontent.com/41194383/161417025-d79f092b-1741-4715-ae99-c237ae4220e4.png)

Clicking on "add filter" will add a subwidget to filter the blobs by one feature value.

![grafik](https://user-images.githubusercontent.com/41194383/161417089-3aa23cac-1194-4bc8-93c7-73cad02b4efb.png)

It may be that the slider of is squished on your screen. In that case, drag the widget window to the left to make it larger:

<video src="https://user-images.githubusercontent.com/41194383/161418489-10110d56-289d-4ed8-86a8-bebdbbe30064.mp4" controls="controls" style="max-width: 500px;">
</video>

Dragging the slider will now filter out blob that are either below the threshold (min) or above the threshold (max). To remove the filter again, you can click "delete". You can add multiple filters that will all be applied to the blobs if you want to filter by multiple features:

![grafik](https://user-images.githubusercontent.com/41194383/161418585-e6a98ee3-7a65-417d-84ef-4b0693ebda7a.png)

To save the filtered blobs and their feature measurements click on "Select file", and choose a filename ending with .csv. Then click on "save results".

### selection widget

Sometimes it can be tricky to find the right parameters for blob detection or the right filters and thresholds to use. In those cases, it can help to use the selection widget. This widget allows you to annotate coordinates manually as true positive (TP) blobs and false positive (FP) blobs. Under the hood, the widget will train a support vector machine to then classify the initially detected blobs into TP and FP blobs. For this to work nicely, it is important that the intially detected blobs contain all the TP blobs that are present in the image (which means they will also contain a lot of FP blobs).

Before you can start you need the following things in napari:

<ul>
<li> A points layer containing detected blobs </li>
<li> An image layer in which the blobs were detected </li>
<li> An additional points layer in which you will make your annotations </li>
</ul>

To add another points layer click on this button in your layer list:

![grafik](https://user-images.githubusercontent.com/41194383/161418957-af8560f9-542b-4b52-8a8b-c6d355cab56d.png)

To open the selection widget go to "Plugins -> napari-blob-detection -> selection_widget". This will dock the selection widget to the napari viewer.

![grafik](https://user-images.githubusercontent.com/41194383/161418807-994bbbee-6e29-4993-9207-7cb286308179.png)

Now make sure to set all the layers correctly:

<ul>
<li> annotation layer: points layer in which you will make your annotations </li>
<li> img layer: image layer on which your annotations will be measured </li>
<li> points layer: points layer in which your detected blobs are present </li>
<li> clf img layer: image layer on which your blobs were detected </li>
</ul>

Then click on "initialize layers". Now you can make annotations on your annotation points layer. You can switch between the "foreground" and "background" class to annotate TP and FP coordinates. The size of the annotations is set automatically based on the size of the blobs in your points layer.

Once you are done adding annotations click the following buttons (in this order!):

<ol>
<li> add training data </li>
<li> Run </li>
<li> apply classifier </li>
</ol>

After clicking "apply classifier" a new points layer will be generated that will only contain the blobs that were classified as TP blobs.

### tracking widget

If you are doing blob detection on 2DT or 3DT images, you may want to track the blobs over time. This is possible with this widget. This widget is an interface to the <a href="http://soft-matter.github.io/trackpy/dev/generated/trackpy.link.html">Trackpy link</a> function. To add the widget, go to "Plugins -> napari-blob-detection -> tracking_widget". This will dock the widget to the napari viewer.

![grafik](https://user-images.githubusercontent.com/41194383/161419611-09dcdcb9-5423-4021-90bb-4832983b7841.png)

<ul>
<li> points layer: points layer containing the detected blobs </li>
<li> search range: range in pixels for which the algorithm will try to link blobs between timepoints </li>
<li> memory: maximum number of timepoints for which a blob can disappear and still be linked back to its original track </li>
</ul>

clicking on "track blobs" will try to track the blobs over time and adds a tracks layer to the napari viewer with the results.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/stable/index.html
-->

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-blob-detection" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description. 

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/adrtsc/napari-blob-detection/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/


