"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the ``napari_get_reader`` hook specification, (to create
a reader plugin) but your plugin may choose to implement any of the hook
specifications offered by napari.
see: https://napari.org/docs/plugins/hook_specifications.html

Replace code below accordingly.  For complete documentation see:
https://napari.org/docs/plugins/for_plugin_developers.html
"""
from typing import List, Any, Dict, Tuple

import numpy as np
import toolz.curried as tz
import os
import dask.array as da

from napari_plugin_engine import napari_hook_implementation
import zarr
from tqdm import tqdm
from skimage import util
from skimage.transform import pyramid_gaussian
from collections import defaultdict
from numcodecs.blosc import Blosc
import json

DOWNSCALE = 2
COLORMAP_HEX_COLOR_DICT = {
    'red': 'FF0000',
    'blue': '0000FF',
    'green': '00FF00'
}

@napari_hook_implementation
def napari_get_writer(path, layer_types):
    """Return ome-zarr writer_function if layer_types are all images.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.
    layer_types: list of str

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path and the layer data
    """
    if all(layer_type in {'image', 'labels'} for layer_type in layer_types):
        return to_ome_zarr
    else:
        return None


def to_ome_zarr(path, layer_data: List[Tuple[Any, Dict, str]]):
    """Save layers to ome-zarr format.

    Take path and list of (data, meta, layer_type) and save to ome-zarr format
    
    Parameters
    ----------
    path : str
        Path to save to

    Returns
    -------
    paths : list of str or None
        List of saved filenames if saving was successful, otherwise None
    """
    # remove the low resolution quick-look image
    layer_data = list(filter(lambda dat: dat[1]['name'] != 'QKL_ALL', layer_data))
    # TODO: because masks are in there we still get two shapes even if no low res bands were selected
    by_shape = tz.groupby(lambda dat: dat[0].shape, layer_data)
    if len(by_shape) > 1:
        basepath, extension = os.path.splitext(path)
        paths = [basepath + '_' + str(shape) + extension for shape in by_shape]
    else:
        paths = [path]

    bands = [layer[1]['name'] for layer in layer_data]
    contrast_histogram = dict(zip(
        bands,
        [[] for i in range(len(bands))]
    ))

    for path, (shape, datasets) in zip(paths, by_shape.items()):
        # only take into account pixels that are in the satellite FOV
        mask = [data[0] for data in datasets if data[1]['name'].startswith('EDG')][0]
        # add a spurious z axis because ome-zarr requires it
        image_layers = [data for data in datasets if data[2] == 'image']
        os.makedirs(path, exist_ok=False)
        out_path = path
        out_zarrs = out_path

        # process each timepoint and band
        for j in tqdm(range(shape[0]), desc=f'writing shape={shape}'):
            for k, (image, image_meta, _) \
                    in tqdm(enumerate(image_layers), desc=f'writing bands'):
                # get downsampled zarr cube for this timepoint and band
                imagej = np.asarray(image[j])
                out_zarrs = band_at_timepoint_to_zarr(
                    imagej,
                    j,
                    k,
                    out_zarrs=out_zarrs,
                    num_timepoints=shape[0],
                    num_bands=len(image_layers),
                )

                # get frequencies of each pixel for this band and timepoint, masking partial tiles
                band_at_timepoint_histogram = get_masked_histogram(
                    imagej, np.asarray(mask[0, :, :])
                )
                band = image_meta['name']
                contrast_histogram[band].append(
                    band_at_timepoint_histogram
                )
            num_resolution_levels = len(out_zarrs)

        contrast_limits = {}
        bands = [image_meta['name'] for _, image_meta, _ in image_layers]
        for band in bands:
            lower, upper = get_contrast_limits(contrast_histogram[band])
            if upper - lower == 0:
                upper += 1 if upper >= 0 else -1
            contrast_limits[band] = (lower, upper)

        band_tuples = [
            (image_meta['name'], COLORMAP_HEX_COLOR_DICT[image_meta['colormap']])
            for _, image_meta, _ in image_layers if
            image_meta['name'] in ['FRE_B2', 'FRE_B3', 'FRE_B4']
        ]
        zattrs = generate_zattrs(
            tile=os.path.basename(path),
            bands=bands,
            contrast_limits=contrast_limits,
            max_layer=num_resolution_levels,
            band_colormap_tup=band_tuples,
        )
        write_zattrs(zattrs, out_path)
    return paths

def band_at_timepoint_to_zarr(
        image,
        timepoint_number,
        band_number,
        *,
        out_zarrs=None,
        min_level_shape=(1024, 1024),
        num_timepoints=None,
        num_bands=None,
):
    """Takes the input timepoint filename and band information and writes the data
    in this image to the appropriate indices in out_zarrs base on the downscale
    If out_zarrs is a string, this is the first iteration, so the appropriate directories are 
    instantiated.

    Parameters
    ----------
    image: dask ndarray
        image to save
    timepoint_number: int
        number of current timepoint in sequence being processed
    band_number : int
        number of current band in sequence being processed
    out_zarrs : string or list, optional
        string path to where zarrs will be stored if first iteration, otherwise list of partially filled pyramid levels, by default None
    min_level_shape : tuple of ints, optional
        shape of smallest desired downscaled level, by default (1024, 1024)
    num_timepoints : int
        total number of timepoints that will be processed, by default None
    num_bands : int
        total number of bands that will be processed, by default None

    Returns
    -------
    list
        partially (or fully) populated list of numpy arrays where each element is one pyramid level
    """
    shape = image.shape
    dtype = image.dtype
    max_layer = np.log2(
        np.max(np.array(shape) / np.array(min_level_shape))
    ).astype(int)
    pyramid = pyramid_gaussian(image, max_layer=max_layer, downscale=DOWNSCALE)
    im_pyramid = list(pyramid)
    if isinstance(out_zarrs, str):
        fout_zarr = out_zarrs
        out_zarrs = []
        compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE, blocksize=0)
        for i in range(len(im_pyramid)):
            r, c = im_pyramid[i].shape
            out_zarrs.append(zarr.open(
                    os.path.join(fout_zarr, str(i)), 
                    mode='a', 
                    shape=(num_timepoints, num_bands, 1, r, c), 
                    dtype=np.int16,
                    chunks=(1, 1, 1, *min_level_shape), 
                    compressor=compressor,
                )
            )

    # for each resolution:
    for pyramid_level, downscaled in enumerate(im_pyramid):
        # convert back to int16
        downscaled = util.img_as_int(downscaled)
        # store into appropriate zarr
        out_zarrs[pyramid_level][timepoint_number, band_number, 0, :, :] = downscaled
    
    return out_zarrs
    

def generate_zattrs(
            tile,
            bands,
            *,
            contrast_limits=None,
            max_layer=5,
            band_colormap_tup=None,
    ) -> Dict:
    """Return a zattrs dictionary matching the OME-zarr metadata spec [1]_.

    Parameters
    ----------
    tile : str
        The input tile name, e.g. "55HBU"
    bands : list of str
        The bands being written to the zarr.
    contrast_limits : dict[str -> (int, int)], optional
        Dictionary mapping bands to contrast limit values.
    max_layer : int
        The highest layer in the multiscale pyramid.
    band_colormap_tup : tuple[(band, hexcolor)]
        List of band to colormap pairs containing all bands to be initially displayed.

    Returns
    -------
    zattr_dict: dict
        Dictionary of OME-zarr metadata.
        
    References
    ----------
    .. [1] https://github.com/ome/omero-ms-zarr/blob/master/spec.md
    """
    band_colormap = defaultdict(lambda: 'FFFFFF', dict(band_colormap_tup))
    zattr_dict = {}
    zattr_dict['multiscales'] = []
    zattr_dict['multiscales'].append({'datasets' : []})
    for i in range(max_layer):
        zattr_dict['multiscales'][0]['datasets'].append(
            {'path': f'{i}'}
        )
    zattr_dict['multiscales'][0]['version'] = '0.1'

    zattr_dict['omero'] = {'channels' : []}
    for band in bands:
        color = band_colormap[band]
        zattr_dict['omero']['channels'].append({
            'active' : band in ['FRE_B2', 'FRE_B3', 'FRE_B4'],
            'coefficient': 1,
            'color': color,
            'family': 'linear',
            'inverted': 'false',
            'label': band,
            'name': band
        })
        if contrast_limits is not None and band in contrast_limits:
            lower_contrast_limit, upper_contrast_limit = contrast_limits[band]
            zattr_dict['omero']['channels'][-1]['window'] = {
                    'end': int(upper_contrast_limit),
                    'max': np.iinfo(np.int16).max,
                    'min': np.iinfo(np.int16).min,
                    'start': int(lower_contrast_limit),
            }
    zattr_dict['omero']['id'] = str(0)
    zattr_dict['omero']['name'] = tile
    zattr_dict['omero']['rdefs'] = {
        'defaultT': 0,  # First timepoint to show the user
        'defaultZ': 0,  # First Z section to show the user
        'model': 'color',  # 'color' or 'greyscale'
    }
    zattr_dict['omero']['version'] = '0.1'
    return zattr_dict


def write_zattrs(zattr_dict, outdir, *, exist_ok=False):
    """Write a given zattr_dict to the corresponding directory/file.

    Parameters
    ----------
    zattr_dict : dict
        The zarr attributes dictionary.
    outdir : str
        The output zarr directory to which to write.
    exist_ok : bool, optional
        If True, any existing files will be overwritten. If False and the
        file exists, raise a FileExistsError. Note that this check only
        applies to .zattrs and not to .zgroup.
    """
    outfile = os.path.join(outdir, '.zattrs')
    if not exist_ok and os.path.exists(outfile):
        raise FileExistsError(
            f'The file {outfile} exists and `exists_ok` is set to False.'
        )
    with open(outfile, "w") as out:
        json.dump(zattr_dict, out)
    
    with open(outdir + "/.zgroup", "w") as outfile:
        json.dump({"zarr_format": 2}, outfile)


def get_masked_histogram(im, mask):
    """Compute histogram of frequencies of each pixel value in the given image, masked to exclude blank areas 
    of partially captured tiles

    Parameters
    ----------
    im : np.ndarray
        uint16 image to mask and compute frequencies.
    mask: np.ndarray
        int labels where 0 is keep and 1 is discard.

    Returns
    -------
    np.ndarray
        histogram of pixel value frequencies for the masked image
    """
    
    # invert to have 0-discard 1-keep
    mask_boolean = np.invert(mask.astype("bool"))

    ravelled = im[mask_boolean]
    masked_histogram = np.histogram(
            ravelled, bins=np.arange(-2**15 - 0.5, 2**15)
        )[0]
    return masked_histogram


def get_contrast_limits(band_frequencies):
    """Compute contrast limits of the given band based on the frequencies of
    pixel values given. Returns the middle 95th percentile.

    Parameters
    ----------
    band_frequencies : list of np.ndarray
        list of pixel value counts for each timepoint processed for this band

    Returns
    -------
    tuple (int, int)
        lower and upper contrast limits for this histogram based on middle 95th percentile
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
