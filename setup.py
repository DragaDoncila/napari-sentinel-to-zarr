#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from setuptools import setup, find_packages


def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding='utf-8').read()


# Add your dependencies in requirements.txt
# Note: you can add test-specific requirements in tox.ini
requirements = []
with open('requirements.txt') as f:
    for line in f:
        stripped = line.split("#")[0].strip()
        if len(stripped) > 0:
            requirements.append(stripped)


# https://github.com/pypa/setuptools_scm
use_scm = {"write_to": "sentinel_to_zarr/_version.py"}

setup(
    name='napari-sentinel-to-zarr',
    version='0.0.5',
    author='Draga Doncila Pop',
    author_email='ddon0001@student.monash.edu',
    license='MIT',
    url='https://github.com/DragaDoncila/napari-sentinel-to-zarr',
    description='Writer plugin for napari to save Sentinel tiffs into ome-zarr format',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Testing',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
    ],
    entry_points={
        'napari.plugin': [
            'napari_sentinel_to_zarr = sentinel_to_zarr.napari_sentinel_to_zarr',
        ],
        'console_scripts': [
            'sentinel-to-zarr = sentinel_to_zarr.sentinel_to_zarr:main'
        ]
    },
)
