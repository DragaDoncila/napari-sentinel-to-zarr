# napari-sentinel-to-zarr

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/napari-sentinel-to-zarr.svg?color=green)](https://pypi.org/project/napari-sentinel-to-zarr)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-sentinel-to-zarr.svg?color=green)](https://python.org)
[![tests](https://github.com/DragaDoncila/napari-sentinel-to-zarr/workflows/tests/badge.svg)](https://github.com/DragaDoncila/napari-sentinel-to-zarr/actions)
<!-- [![codecov](https://codecov.io/gh/DragaDoncila/napari-sentinel-to-zarr/branch/master/graph/badge.svg)](https://codecov.io/gh/DragaDoncila/napari-sentinel-to-zarr) -->

Writer plugin for napari to save Sentinel tiffs into ome-zarr format

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/docs/plugins/index.html
-->

## Installation

You can install `napari-sentinel-to-zarr` via [pip]:

    `pip install napari-sentinel-to-zarr`


## Usage
This package provides command-line utilities for:
- Processing raw Sentinel zips of a tile into multiscale zarr
- Processing interpolated “GapFilled” tif for that tile into zarr, rechunking along time axis
- Processing the zarr from step 2 into multiscale OME-zarr


### Raw Zipped Images to Zarr
Install the latest version of napari using

`pip install -U napari`

Install napari-sentinel-to-zarr using 

`pip install napari-sentinel-to-zarr`

Run `sentinel-to-zarr zip-to-zarr path/to/tile/55HBU out/path/dir/55HBU.zarr`

where `path/to/tile/55HBU` is a directory full of Sentinel zips

### Interpolated Images to Zarr
Run `sentinel-to-zarr interpolated-to-zarr path/to/interpolated/55HBU_GapFilled_Image.tif out/dir/Image.zarr`

You can pass in optional arguments:
- `--step-size` - how many slices to convert at once. The default is 20 which will require ~5GB RAM. A larger step size means more slices can be loaded at once, and will speed up performance. A full image is typically ~175GB.
- `--chunk-size` - the chunk size of the resulting zarr. Default is 1024 which is typically considered a good option and provides good performance


### Interpolated Zarr to Multiscale Zarr
Run `sentinel-to-zarr zarr-to-multiscale-zarr path/to/interpolated/zarr out/path/interpolated_multiscale.zarr tilename`

Where `path/to/interpolated/zarr` is the output file from step 2

You can pass in optional arguments
- `--min-shape-` the smallest pyramid level you wish to generate. Default is 1024px

## Napari Plugin
This package also comes with a napari writer plugin which allows you to open the raw tiles directly by dragging one into a napari viewer, and saving out while you browse using `File>Save Layers`

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-sentinel-to-zarr" is free and open source software

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
[file an issue]: https://github.com/DragaDoncila/napari-sentinel-to-zarr/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/