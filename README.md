# napari-blob-detection

[![License](https://img.shields.io/pypi/l/napari-blob-detection.svg?color=green)](https://github.com/adrtsc/napari-blob-detection/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-blob-detection.svg?color=green)](https://pypi.org/project/napari-blob-detection)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-blob-detection.svg?color=green)](https://python.org)
[![tests](https://github.com/adrtsc/napari-blob-detection/workflows/tests/badge.svg)](https://github.com/adrtsc/napari-blob-detection/actions)
[![codecov](https://codecov.io/gh/adrtsc/napari-blob-detection/branch/master/graph/badge.svg)](https://codecov.io/gh/adrtsc/napari-blob-detection)

napari plugin for blob detection in images. Implements a user interface to use scikit-image blob detection algorithms (https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_blob.html) and an additional widget to filter the detected blobs by a blob feature value (or a combination of feature values).


https://user-images.githubusercontent.com/41194383/120099217-b2fb2700-c13a-11eb-9e9a-1c00eaa114c0.mp4


----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/docs/plugins/index.html
-->

## Installation

You can install `napari-blob-detection` via [pip]:

    pip install git+https://github.com/adrtsc/napari-blob-detection.git
    
You will need a recent version of magicgui with some bugfixes for proper visualization:

    pip install git+https://github.com/napari/magicgui.git
    
## Examples

If you would like to try the examples make sure that you have jupyter notebook installed and do the following:

    git clone https://github.com/adrtsc/napari-blob-detection
    cd napari-blob-detection/examples/
    
Then, if you want to try the 2D example:

    jupyter notebook blob_detection_example.ipynb
    
 And for the 3D example:
 
    jupyter notebook blob_detection_example_3D.ipynb



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
