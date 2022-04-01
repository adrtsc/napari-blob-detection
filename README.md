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

https://user-images.githubusercontent.com/41194383/120099217-b2fb2700-c13a-11eb-9e9a-1c00eaa114c0.mp4

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

If you want to use the plugin with your own data, it is important that you convert your image into a 4D (t, z, y, x) array before adding it as image layer to napari. Otherwise the plugin will struggle.
    
    
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
