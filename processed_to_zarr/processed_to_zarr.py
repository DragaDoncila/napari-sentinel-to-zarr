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

CHANNELS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12"
]
TOTAL_CHANNELS = 10
DOWNSCALE = 2

parser = argparse.ArgumentParser()
parser.add_argument(
    'root_path',
    help='The root path containing interpolated image and mask for one tile.',
)
parser.add_argument(
    'out_path',
    help='An output directory to which to write output .zarr files.',
)
parser.add_argument(
    '--chunk_size',
    default=1024,
    help='Chunk size of output zarr files.',
    dest='chunk_size'
)
parser.add_argument(
    '--step_size',
    default=20,
    help='Number of 10980*10980 slices to convert at once.',
    dest='step_size'
)
parser.add_argument(
    '--min_shape',
    default = (1024, 1024),
    help="Smallest resolution of multiscale pyramid",
    dest="min_shape"
)

def main(argv=sys.argv[1:]):
    args = parser.parse_args(argv)
    path_dir = args.root_path

    tilename = os.path.basename(os.path.normpath(path_dir))
    #TODO: better way to concatenate the paths?
    fn = path_dir + tilename + "_GapFilled_Image.tif"
    out_fn = args.out_path + tilename + "_GapFilled_Image.zarr"
    
    # write out all channels to single scale zarr
    processed_im_to_rechunked_zarr(fn, out_fn, args.chunk_size, args.step_size)
        
    # process into multiscale and reshape to pull out channels
    #TODO: could we delete as we go somehow to save room?
    multiscale_fn = args.out_path + tilename + "_Multiscale_GapFilled_Image.zarr"
    
    max_layer, contrast_histogram = zarr_to_multiscale_zarr(out_fn, multiscale_fn, args.min_shape, args.chunk_size)
    # compute contrast limits and write zattrs
    contrast_limits = {}
    for band in CHANNELS:
        lower, upper = get_contrast_limits(contrast_histogram[band])
        contrast_limits[band] = (lower, upper)

    write_zattrs(multiscale_fn, contrast_limits, max_layer, tilename)

def processed_im_to_rechunked_zarr(filename, outname, chunk_size, step_size):
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
                chunks=(1, chunk_size, chunk_size), 
                compressor=compressor
                )

    start = 0
    end = 0
    num_chunks = z_arr.shape[0] // step_size

    for i in tqdm(range(0, num_chunks)):
        start = i * step_size
        end = start + step_size
        current_slice = np.copy(d_transposed[start:end, :, :])
        z_arr[start:end, :, :] = current_slice
        del(current_slice)

    print("Copying remainder...")
    if z_arr.shape[0] % step_size != 0:
        final_slice = np.copy(d_transposed[end:, :, :])
        z_arr[end:, :, :] = final_slice


def zarr_to_multiscale_zarr(fn, out_fn, min_level_shape, chunk_size):
    im = da.from_zarr(fn)

    im_shape = im.shape
    num_slices = im_shape[0] // TOTAL_CHANNELS
    x = im_shape[1]
    y = im_shape[2]
    im = da.reshape(im, (num_slices, TOTAL_CHANNELS, x, y))
    compressor = Blosc(cname='zstd', clevel=9, shuffle=Blosc.SHUFFLE, blocksize=0)
    max_layer = np.log2(
        np.max(np.array(im_shape[1:]) / np.array(min_level_shape))
    ).astype(int)

    contrast_histogram = dict(zip(
        CHANNELS,
        [[] for i in range(TOTAL_CHANNELS)]
    ))

    Path(out_fn).mkdir(parents=True, exist_ok=True)
    # open zarr arrays for each resolution shape (num_slices, res, res)
    zarrs = []
    for i in range(max_layer+1):
        new_res = tuple(np.ceil(np.array(im_shape[1:]) / (DOWNSCALE ** i)).astype(int)) if i != 0 else im_shape[1:]
        outname = out_fn + f"/{i}"

        z_arr = zarr.open(
                outname, 
                mode='w', 
                shape=(num_slices, TOTAL_CHANNELS, 1, new_res[0], new_res[1]), 
                dtype=im.dtype,
                chunks=(1, 1, 1, chunk_size, chunk_size), 
                compressor=compressor
                )
        zarrs.append(z_arr)

    # for each slice
    for i in tqdm(range(num_slices)):
        for j in tqdm(range(TOTAL_CHANNELS)):
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
    
    return max_layer, contrast_histogram


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


def write_zattrs(out_fn, contrast_limits, max_layer, tilename):
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
    for band in CHANNELS:
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


