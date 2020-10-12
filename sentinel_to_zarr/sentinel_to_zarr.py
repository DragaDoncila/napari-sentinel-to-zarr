import argparse
import os
import sys
import tifffile
from numcodecs.blosc import Blosc
import zarr
import numpy as np
from tqdm import tqdm
import json
import functools
import operator
import skimage
from skimage.transform import pyramid_gaussian
from pathlib import Path
import dask.array as da
import itertools
from sentinel_to_zarr.napari_sentinel_to_zarr import to_ome_zarr
from napari_sentinel_zip.napari_sentinel_zip import reader_function
from collections import defaultdict
import napari
from ast import literal_eval
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure

INTERPOLATED_BANDS_LIST = [
    "FRE_B2",
    "FRE_B3",
    "FRE_B4",
    "FRE_B5",
    "FRE_B6",
    "FRE_B7",
    "FRE_B8",
    "FRE_B8A",
    "FRE_B11",
    "FRE_B12" 
]

DOWNSCALE = 2
BAND_HEX_COLOR_DICT = {
    'FRE_B2': 'FF0000',
    'FRE_B3': '0000FF',
    'FRE_B4': '00FF00'
}

LABEL_MAPPING = "./sentinel_to_zarr/class_map.txt"
DEFAULT_LIMS = (-20, 1900)

def zip_to_zarr(args):
    """Save raw, zipped Sentinel tiffs to multiscale OME-zarr

    Parameters
    ----------
    args : argparse Namespace
        expecting path to directory of Sentinel zips and path to output zarr
    """
    data = reader_function(args.root_path)
    to_ome_zarr(args.out_zarr, data)


def interpolated_to_zarr(args):
    """Write interpolated Sentinel image to zarr, transposing axes to time oriented

    Parameters
    ----------
    args : argparse Namespace
        Expecting input tiff path and output zarr attributes
    """
    fn = args.in_tif
    out_fn = args.out_zarr
    
    processed_im_to_rechunked_zarr(fn, out_fn, args.chunk_size, args.step_size)


def zarr_to_multiscale_zarr(args):
    """Write input zarr to multiscale OME-zarr with separate channel axis

    Parameters
    ----------
    args : argparse Namespace
        Expecting input zarr path and output zarr attributes
    """
    fn = args.in_zarr 
    out_fn = args.out_zarr     
    min_level_shape = (args.min_shape, args.min_shape)
    bands = INTERPOLATED_BANDS_LIST

    write_multiscale_zarr(fn, out_fn, min_level_shape, bands, args.tilename)

def load_all(args):
    """Load raw, interpolated and label OME-zarrs into napari

    Parameters
    ----------
    args : argparse Namespace
        We expect raw_path as path to raw Sentinel image OME-zarr, interpolated_path as path to 
        interpolated Sentinel image OME-zarr, and labels_path as path to label tiff or single-scale zarr.
    """
    LEVEL = 0
    label_properties, colour_dict = get_label_properties()

    with napari.gui_qt():
        napari.utils.dask_utils.resize_dask_cache(mem_fraction=0.1)
        viewer = napari.Viewer()
        viewer.open(
            args.raw_path,
            scale=(365/108, 1, 1, 1),
            visible=False
            )
        for layer in viewer.layers:
            layer.name = f"Raw_{layer.name}"
        viewer.open(
            args.interpolated_path,
            scale=(365/73, 1, 1, 1), 
            multiscale=True,
        )
        viewer.open(
            args.labels_path,
            scale=(1,1),
            layer_type="labels",
            properties=label_properties,
            color=colour_dict,
            opacity=0.4
        )

        # aribitrarily grab red layer
        intensity_plot_im = viewer.layers['FRE_B2'].data[LEVEL]
        # create the intensity plot
        with plt.style.context('dark_background'):
            intensity_canvas = FigureCanvas(Figure(figsize=(5, 3)))
            intensity_axes = intensity_canvas.figure.subplots()
            intensities = intensity_plot_im[:, 0, 0, 0]

            intensity_line = intensity_axes.plot(np.arange(intensity_plot_im.shape[0]), intensities)[0]  # returns line list
            position_line = intensity_axes.axvline(x=0, c='C1')
            position_line.set_zorder(-1)  # keep the spectra in front
            minval, maxval = np.min(intensities), np.max(intensities)
            range_ = maxval - minval
            centre = (maxval + minval) / 2
            min_y = centre - 1.05 * range_ / 2
            max_y = centre + 1.05 * range_ / 2
            intensity_axes.set_ylim(min_y, max_y)
            intensity_axes.set_xlabel('time')
            intensity_axes.set_ylabel('intensity')
            title = intensity_axes.set_title('Intensity at: coord=(0, 0)')
            intensity_canvas.figure.tight_layout()


        # add the plot to the viewer
        viewer.window.add_dock_widget(intensity_canvas)


        # create a function to update the plot
        def update_plot(axis_event):
            axis = axis_event.axis
            if axis != 0:
                return
            x = axis_event.value
            position_line.set_data([x, x], [0, 1])
            intens_str, coord_str = title.get_text().split(':')
            title.set_text(intens_str + ":" + coord_str)
            intensity_canvas.draw_idle()


        # connect the function to the dims axis
        viewer.dims.events.axis.connect(update_plot)


        def update_intensity(layer, event):
            xs, ys = intensity_line.get_data()
            coords_full = tuple(np.round(layer._transforms.simplified(layer.coordinates)).astype(int) // 2**LEVEL) 
            if "LABELS" in layer.name.upper():
                in_range = all(coords_full[i] in range(intensity_plot_im.shape[i+2]) for i in range(2))
                coords = tuple((0, *coords_full))   # z, rows, columns
            else:
                in_range = all(coords_full[i] in range(intensity_plot_im.shape[i])
                    for i in range(intensity_plot_im.ndim))
                coords = coords_full[1:]  # z, rows, columns
            if in_range:
                print(f"Updating plot for {coords_full[-2::]}...")
                new_ys = intensity_plot_im[:, coords[0], coords[1], coords[2]]
                min_y = np.min(new_ys)
                max_y = np.max(new_ys)
                intensity_axes.set_ylim(min_y, max_y)
                intensity_line.set_data(xs, new_ys)
                intens_str, coords_str = title.get_text().split(':')
                title.set_text(intens_str + ': ' + str(coords))
                intensity_canvas.draw_idle()

        for layer in viewer.layers:
            # add a click callback to each layer to update the spectrum being viewed
            layer.mouse_drag_callbacks.append(update_intensity)

            
def get_label_properties():
    df = pd.read_csv(LABEL_MAPPING)

    dicts = df.to_dict('split')
    classes = list(df['class'])
    colors = [tuple([v / 255 for v in literal_eval(val)]) for val in list(df['colour'])]

    label_properties = {
        'class': ['None'] + classes
    }

    colour_indices = [i for i in range(df.shape[0] + 1)]
    colours = [(0, 0, 0, 0)] + colors
    colour_dict = dict(zip(colour_indices, colours))

    return label_properties, colour_dict


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers()

parser_zip_to_zarr = subparsers.add_parser('zip-to-zarr')
parser_zip_to_zarr.add_argument(
    'root_path',
    help='The root path containing raw Sentinel zips e.g. ~/55HBU/',
)
parser_zip_to_zarr.add_argument(
    'out_zarr',
    help='Path to output zarr file e.g. ~/55HBU.zarr'
)
parser_zip_to_zarr.set_defaults(
    func=zip_to_zarr
)

parser_interpolated_to_zarr = subparsers.add_parser('interpolated-to-zarr')
parser_interpolated_to_zarr.add_argument(
    'in_tif',
    help='The path to interpolated Sentinel tif for one tile',
)
parser_interpolated_to_zarr.add_argument(
    'out_zarr',
    help='Path to output zarr file e.g. ~/55HBU_GapFilled_Image.zarr'
)
parser_interpolated_to_zarr.add_argument(
    '--chunk-size',
    default=1024,
    help='Chunk size of output zarr files.',
    dest='chunk_size'
)
parser_interpolated_to_zarr.add_argument(
    '--step-size',
    default=20,
    help='Number of 10980*10980 slices to convert at once.',
    dest='step_size'
)
parser_interpolated_to_zarr.set_defaults(
    func=interpolated_to_zarr
)

parser_zarr_to_multiscale = subparsers.add_parser('zarr-to-multiscale-zarr')
parser_zarr_to_multiscale.add_argument(
    'in_zarr',
    help='Path to interpolated zarr e.g. ~/55HBU_GapFilled_Image.zarr',
)
parser_zarr_to_multiscale.add_argument(
    'out_zarr',
    help='Path to output directory for zarr file e.g. ~/55HBU_Multiscale.zarr'
)
parser_zarr_to_multiscale.add_argument(
    'tilename',
    help='Name of tile currently being processed'
)
parser_zarr_to_multiscale.add_argument(
    '--min-shape',
    default =1024,
    type=int,
    help="Smallest resolution of multiscale pyramid",
    dest="min_shape"
)
parser_zarr_to_multiscale.set_defaults(
    func= zarr_to_multiscale_zarr
)

parser_load_all = subparsers.add_parser('load-all')
parser_load_all.add_argument(
    'raw_path',
    help='Path to OME-zarr of raw images',
)
parser_load_all.add_argument(
    'interpolated_path',
    help='Path to OME-zarr of interpolated images'
)
parser_load_all.add_argument(
    'labels_path',
    help='Path to labels tiff'
)
parser_load_all.set_defaults(
    func=load_all
)

def main(argv=sys.argv[1:]):
    args = parser.parse_args(argv)
    args.func(args)


def processed_im_to_rechunked_zarr(filename, outname, chunk_size, step_size):
    """Write interpolated Sentinel image to zarr, transposing axes to time oriented

    Parameters
    ----------
    filename : str
        path to interpolated tif
    outname : str
        path to output zarr
    chunk_size : int
        desired chunk_size for visible axes
    step_size : int
        number of 10980*10980 slices to process at once
    """
    tiff_f = tifffile.TiffFile(filename)
    d_mmap = tiff_f.pages[0].asarray(out='memmap')
    tiff_f.close()
    d_transposed = d_mmap.transpose((2, 1, 0))

    compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE, blocksize=0)

    out_zarr = zarr.open(
            outname, 
            mode='w', 
            shape=d_transposed.shape, 
            dtype=d_transposed.dtype,
            chunks=(1, chunk_size, chunk_size), 
            compressor=compressor
            )

    for start in tqdm(range(0, d_transposed.shape[0], step_size)):
        end = min(start + step_size, d_transposed.shape[0])

        current_slice = np.copy(d_transposed[start:end, :, :])
        out_zarr[start:end, :, :] = current_slice
        del(current_slice)


def write_multiscale_zarr(fn, out_fn, min_level_shape, bands, tilename):
    """Write input zarr to multiscale OME-zarr with separate channel axis

    Parameters
    ----------
    fn : str
        path to input zarr
    out_fn : str
        path to output zarr
    min_level_shape : tuple (int, int)
        smallest desired resolution
    bands : list
        list of band names being processed
    tilename : str
        name of tile being processed
    """
    im = da.from_zarr(fn)

    im_shape = im.shape
    num_slices = im_shape[0] // len(bands)
    x = im_shape[1]
    y = im_shape[2]
    im = da.reshape(im, (num_slices, len(bands), x, y))
    compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE, blocksize=0)
    max_layer = np.log2(
        np.max(np.array(im_shape[1:]) / np.array(min_level_shape))
    ).astype(int)

    contrast_histogram = dict(zip(
        bands,
        [[] for i in range(len(bands))]
    ))

    Path(out_fn).mkdir(parents=True, exist_ok=False)
    # open zarr arrays for each resolution shape (num_slices, res, res)
    zarrs = []
    for i in range(max_layer+1):
        new_res = tuple(np.ceil(np.array(im_shape[1:]) / (DOWNSCALE ** i)).astype(int)) if i != 0 else im_shape[1:]
        outname = out_fn + f"/{i}"

        z_arr = zarr.open(
                outname, 
                mode='w', 
                shape=(num_slices, len(bands), 1, new_res[0], new_res[1]), 
                dtype=im.dtype,
                chunks=(1, 1, 1, im.chunksize[2], im.chunksize[3]), 
                compressor=compressor
                )
        zarrs.append(z_arr)
    # for each slice
    for i in tqdm(range(num_slices)):
        for j in tqdm(range(len(bands))):
            im_slice = im[i, j, :, :]
            # get pyramid
            im_pyramid = list(pyramid_gaussian(im_slice, max_layer=max_layer, downscale=DOWNSCALE))
            # for each resolution
            for k, new_im in enumerate(im_pyramid):
                # convert to uint16
                new_im = skimage.img_as_uint(new_im)
                # store into appropriate zarr at (slice, band, :)
                zarrs[k][i, j, 0, :, :] = new_im
            contrast_histogram[bands[j]].append(
                get_histogram(new_im)
            )
    
    contrast_limits = {}
    for band in bands:
        lower, upper = get_contrast_limits(contrast_histogram[band])
        if upper - lower == 0:
            lower, upper = DEFAULT_LIMS
        contrast_limits[band] = (lower, upper)

    write_zattrs(out_fn, contrast_limits, max_layer, tilename, bands)


def get_contrast_limits(band_frequencies):
    """Compute middle 95th percentile contrast limits using given band frequencies

    Parameters
    ----------
    band_frequencies : list of np.ndarray
        list of pixel value counts for each timepoint processed for this band
    Returns
    -------
    tuple (int, int)
        lower and upper 95th percentile contrast limits for this histogram 
    """
    frequencies = sum(band_frequencies)
    lower_limit = np.flatnonzero(
        np.cumsum(frequencies) / np.sum(frequencies) > 0.025
    )[0]
    upper_limit = np.flatnonzero(
        np.cumsum(frequencies) / np.sum(frequencies) > 0.975
    )[0]
    lower_limit_rescaled = lower_limit - 2**15
    upper_limit_rescaled = upper_limit - 2**15
    return lower_limit_rescaled, upper_limit_rescaled


def write_zattrs(out_fn, contrast_limits, max_layer, tilename, bands):
    """Write zattrs dictionary matching the OME-zarr metadata spec [1]_ to file.

    Parameters
    ----------
    out_fn : str
        path to corresponding OME-zarr
    contrast_limits : dict
        dictionary of bands to contrast limits
    max_layer : int
        the highest layer in the multiscale pyramid
    tilename : str
        name of the tile being processed
    bands : list of str
        list of bands being processed

    References
    ----------
    .. [1] https://github.com/ome/omero-ms-zarr/blob/master/spec.md
    """
    band_color_dict = defaultdict(lambda: 'FFFFFF', zip(BAND_HEX_COLOR_DICT.keys(), BAND_HEX_COLOR_DICT.values()))
    # write zattr file with contrast limits and remaining attributes
    zattr_dict = {}
    zattr_dict["multiscales"] = []
    zattr_dict["multiscales"].append({"datasets" : []})
    for i in range(max_layer+1):
        zattr_dict["multiscales"][0]["datasets"].append({
            "path": f"{i}"
        })
    zattr_dict["multiscales"][0]["version"] = "0.1"

    zattr_dict["omero"] = {"channels" : []}
    for band in bands:
        zattr_dict["omero"]["channels"].append(
            {
            "active" : band in BAND_HEX_COLOR_DICT.keys(),
            "coefficient": 1,
            "color": band_color_dict[band],
            "family": "linear",
            "inverted": "false",
            "label": band,
            "name": band,
            "window": {
                "end": int(contrast_limits[band][1]),
                "max": 65535,
                "min": 0,
                "start": int(contrast_limits[band][0])
            }
            }
        )
    zattr_dict["omero"]["id"] = str(0)
    zattr_dict["omero"]["name"] = tilename
    zattr_dict["omero"]["rdefs"] = {
        "defaultT": 0,                    # First timepoint to show the user
        "defaultZ": 0,                  # First Z section to show the user
        "model": "color"                  # "color" or "greyscale"
    }
    zattr_dict["omero"]["version"] = "0.1"

    with open(out_fn + "/.zattrs", "w") as outfile:
        json.dump(zattr_dict, outfile)

    with open(out_fn + "/.zgroup", "w") as outfile:
        json.dump({"zarr_format": 2}, outfile)


def get_histogram(im):
    """Compute histogram of frequencies of each pixel value in the given image

    Parameters
    ----------
    im : np.ndarray
        uint16 image to compute frequencies for

    Returns
    -------
    np.ndarray
        histogram of pixel value frequencies for the image
    """

    ravelled = np.ravel(im)
    histogram = np.histogram(
            ravelled, bins=np.arange(-2**15 - 0.5, 2**15)
        )[0]
    return histogram


def recompute_contrast_limits(in_zarr, pyramid_level):
    """Helper function to recompute contrast limits for a multiscale multichannel OME-zarr

    Parameters
    ----------
    in_zarr : str
        path to OME-zarr
    pyramid_level : int
        level of pyramid to use for contrast limits.
    """
    bands, max_layer, tilename = get_zattr_info(in_zarr)
    in_im = da.from_zarr(f"{in_zarr}/{pyramid_level}")

    contrast_histogram = dict(zip(
        bands,
        [[] for i in range(len(bands))]
    ))

    for i,j in tqdm(itertools.product(range(in_im.shape[0]), range(in_im.shape[1]))):
            current_im = in_im[i, j, 0, :, :]
            current_band = bands[j]

            contrast_histogram[current_band].append(
                get_histogram(np.asarray(current_im))
            )
    
    contrast_limits = {}
    for band in bands:
        lower, upper = get_contrast_limits(contrast_histogram[band])
        if upper - lower == 0:
            lower, upper = DEFAULT_LIMS
        contrast_limits[band] = (lower, upper)

    write_zattrs(in_zarr, contrast_limits, max_layer, tilename, bands)

def get_zattr_info(in_zarr):
    """Parse zattrs of given OME-zarr and return bands, max layer and tilename

    Parameters
    ----------
    in_zarr : str
        path to OME-zarr

    Returns
    -------
    info: tuple(list, int, str)
        tuple of bands, max layer and tilename retrieved from zattrs
    """
    with open(in_zarr+'/.zattrs') as f:
        zattrs = json.load(f)
    bands = [channel['name'] for channel in zattrs['omero']['channels']]
    
    pth_list = zattrs['multiscales'][0]['datasets']
    layers = [int(pth_list[int(i)]['path']) for i in range(len(pth_list))]
    max_layer = max(layers)

    tilename = zattrs['omero']['name']
    return bands, max_layer, tilename
