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
from rechunker import rechunk
from dask.diagnostics import progress
from sentinel_to_zarr.napari_sentinel_to_zarr import to_ome_zarr
from napari_sentinel_zip.napari_sentinel_zip import reader_function


def zip_to_zarr(args):
    data = reader_function(args.root_path)
    data = [layer_data for layer_data in data if layer_data[1]['name'] in args.selected_bands or layer_data[1]['name'].startswith('EDG')]
    to_ome_zarr(args.out_zarr, data)

def interpolated_to_zarr(args):
    fn = args.in_tif
    out_fn = args.out_zarr
    
    # write out all channels to single scale zarr
    processed_im_to_rechunked_zarr(fn, out_fn, args.chunk_size, args.step_size, args.selected_bands)

def zarr_to_multiscale_zarr(args):
    print("zarr to multiscale", args)
    fn = args.in_zarr 
    out_fn = args.out_zarr     
    min_level_shape = args.min_shape
    bands = open(fn + "/bands.txt", 'r').readline().split(',')

    im = da.from_zarr(fn)

    im_shape = im.shape
    num_slices = im_shape[0] // NUM_INTERPOLATED_BANDS
    x = im_shape[1]
    y = im_shape[2]
    im = da.reshape(im, (num_slices, NUM_INTERPOLATED_BANDS, x, y))
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
                shape=(num_slices, NUM_INTERPOLATED_BANDS, 1, new_res[0], new_res[1]), 
                dtype=im.dtype,
                chunks=(1, 1, 1, im.chunks[1], im.chunks[2]), 
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
                print(k, i, j)
                # conver to uint16
                new_im = skimage.img_as_uint(new_im)
                # store into appropriate zarr at (slice, band, :)
                zarrs[k][i, j, 0, :, :] = new_im
            contrast_histogram[j].append(
                get_histogram(im_pyramid[-1])
            )
    
    contrast_limits = {}
    for band in bands:
        lower, upper = get_contrast_limits(contrast_histogram[band])
        contrast_limits[band] = (lower, upper)

    write_zattrs(out_fn, contrast_limits, max_layer, args.tilename, bands)

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
parser_zip_to_zarr.add_argument(
    '--bands',
    help='The bands to process.',
    type=lambda string: string.split(','),
    default= ['FRE_B2', 'FRE_B3', 'FRE_B4', 'FRE_B8'],
    dest='selected_bands'
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
    '--chunk_size',
    default=1024,
    help='Chunk size of output zarr files.',
    dest='chunk_size'
)
parser_interpolated_to_zarr.add_argument(
    '--step_size',
    default=20,
    help='Number of 10980*10980 slices to convert at once.',
    dest='step_size'
)
parser_interpolated_to_zarr.add_argument(
    '--bands',
    help='The bands to process.',
    type=lambda string: string.split(','),
    default= ['FRE_B2', 'FRE_B3', 'FRE_B4', 'FRE_B8'],
    dest='selected_bands'
)
parser_interpolated_to_zarr.set_defaults(
    func=interpolated_to_zarr
)

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
INTERPOLATED_BAND_INDICES = list(range(10))
INTERPOLATED_BANDS_TO_INDICES = dict(zip(INTERPOLATED_BANDS_LIST, INTERPOLATED_BAND_INDICES))
INTERPOLATED_INDICES_TO_BANDS = dict(zip(INTERPOLATED_BAND_INDICES, INTERPOLATED_BANDS_LIST))

NUM_INTERPOLATED_BANDS = 10
DOWNSCALE = 2
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
    '--min_shape',
    default = (1024, 1024),
    help="Smallest resolution of multiscale pyramid",
    dest="min_shape"
)
parser_zarr_to_multiscale.set_defaults(
    func= zarr_to_multiscale_zarr
)


def main(argv=sys.argv[1:]):
    args = parser.parse_args(argv)
    args.func(args)


def processed_im_to_rechunked_zarr(filename, outname, chunk_size, step_size, selected_bands):
    tiff_f = tifffile.TiffFile(filename)
    d_mmap = tiff_f.pages[0].asarray(out='memmap')
    tiff_f.close()
    d_transposed = d_mmap.transpose((2, 0, 1))

    compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE, blocksize=0)

    num_timepoints = d_transposed.shape[0] // 10
    num_selected_slices = (num_timepoints) * len(selected_bands)
    selected_band_indices = sorted([INTERPOLATED_BANDS_TO_INDICES[band] for band in selected_bands])
    out_bands = [INTERPOLATED_INDICES_TO_BANDS[idx] for idx in selected_band_indices]
    with open(outname + "/bands.txt", 'w') as band_file:
        for band in out_bands:
            band_file.write(band + ", ")

    
    selected_band_indices = list(itertools.chain.from_iterable([[i*10 + idx for idx in selected_band_indices] for i in range(num_timepoints)]))


    z_arr = zarr.open(
                outname, 
                mode='w', 
                shape=(num_selected_slices, d_transposed.shape[1], d_transposed.shape[2]), 
                dtype=d_transposed.dtype,
                chunks=(1, chunk_size, chunk_size), 
                compressor=compressor
                )

    start = 0
    end = 0
    num_chunks = num_selected_slices // step_size
    start_zarr = 0
    end_zarr = 0
    for i in tqdm(range(0, num_chunks)):
        start = i * step_size
        end = start + step_size
        current_slice_indices = selected_band_indices[start:end]

        current_slice = np.copy(d_transposed[current_slice_indices, :, :])
        end_zarr += len(current_slice)
        z_arr[start_zarr:end_zarr, :, :] = current_slice
        start_zarr = end_zarr
        del(current_slice)

    print("Copying remainder...")
    if num_selected_slices % step_size != 0:
        final_slice_indices = selected_band_indices[end:]
        final_slice = np.copy(d_transposed[current_slice_indices, :, :])
        z_arr[end_zarr:, :, :] = final_slice


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


def write_zattrs(out_fn, contrast_limits, max_layer, tilename, bands):
    # write zattr file with contrast limits and remaining attributes
    zattr_dict = {}
    zattr_dict["multiscales"] = []
    zattr_dict["multiscales"].append({"datasets" : []})
    for i in range(max_layer):
        zattr_dict["multiscales"][0]["datasets"].append({
            "path": f"{i}"
        })
    zattr_dict["multiscales"][0]["version"] = "0.1"

    zattr_dict["omero"] = {"channels" : []}
    for band in bands:
        zattr_dict["omero"]["channels"].append(
            {
                # TODO: write proper colors and active channels here
            "active" : i==0,
            "coefficient": 1,
            "color": "FFFFFF",
            "family": "linear",
            "inverted": "false",
            "label": band,
            "window": {
                "end": contrast_limits[band][1],
                "max": 65535,
                "min": 0,
                "start": contrast_limits[band][0]
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
    masked_histogram = np.histogram(
            ravelled, bins=np.arange(-2**15 - 0.5, 2**15)
        )[0]
    return masked_histogram



def processed_im_to_zarr(filename, outname, chunk_size, step_size):
    tiff_f = tifffile.TiffFile(filename)
    d_mmap = tiff_f.pages[0].asarray(out='memmap')
    tiff_f.close()
    d_transposed = d_mmap.transpose((2, 0, 1))

    compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE, blocksize=0)

    z_arr = zarr.open(
                outname, 
                mode='w', 
                shape=d_transposed.shape, 
                dtype=d_transposed.dtype,
                chunks=(d_transposed.shape[0], 1, 1000), 
                compressor=compressor
                )

    for i in tqdm(range(d_transposed.shape[1])):
        z_arr[:, i, :] = d_transposed[:, i, :]


def rechunk_zarr(in_fn, out_dir, chunk_size):
    source = da.from_zarr(in_fn)
    intermediate = out_dir + "_intermediate.zarr"
    target = out_dir + "_Rechunked_GapFilled_Image.zarr"
    rechunked = rechunk(
        source, 
        target_chunks=(1, chunk_size, chunk_size), 
        target_store=target,
        max_mem="8GB",
        temp_store=intermediate)
    print(rechunked)
    with progress.ProgressBar():
        rechunked.execute()


