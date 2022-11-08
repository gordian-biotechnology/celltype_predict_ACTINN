#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os
import codecs
import sys

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

'''
with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()
'''
requirements = [
    "numpy>=1.15",
    "anndata>=0.6.22",
    "pandas>=1.3",
    "h5py",
    "scanpy>=1.8",
    "tensorflow>=2.4.0"
]


author = 'Ian Driver'

setup(
    author=author,
    author_email='ian@gordian.bio',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    description="ACTINN based celltype prediction",
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    keywords='single-cell',
    name='celltype_predict_ACTINN',
    packages=find_packages(),
    package_dir={'celltype_predict_ACTINN':
                 'celltype_predict_ACTINN'},
    entry_points={
          'console_scripts': ['celltype_predict_ACTINN = celltype_predict_ACTINN.run_actinn:main']
        },
    version=get_version("celltype_predict_ACTINN/__init__.py"),
    zip_safe=False,
)