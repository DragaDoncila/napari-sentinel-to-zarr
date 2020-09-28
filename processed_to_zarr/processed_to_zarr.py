import argparse
import os
import sys

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
    default=2048,
    help='Chunk size of output zarr files.',
    dest='chunk_size'
)
parser.add_argument(
    '--step_size',
    default=20,
    help='Number of 10980*10980 slices to convert at once.',
    dest='step_size'
)

def main(argv=sys.argv):
    args = parser.parse_args(argv)
    path_dir = args.root_path

    tilename = os.path.basename(path_dir)
    print(path_dir)

