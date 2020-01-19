from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import sys


CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS"
]

extra_compile_args=['-std=c++11']
if sys.platform == 'darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.9']

ext_modules = [
    Extension(
        'acim.wrapper',
        ['cpp/lstm.cpp' , 'cpp/wrapper.cpp'],
        include_dirs=[ pybind11.get_include() ],
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_compile_args,
    ),
]

setup(
    name = 'KWS LSTM',
    packages = [ 'acim' ],
    data_files = [
        ('acim', [ "acim/kws.pickle" ] ),  
    ],
    ext_modules=ext_modules,
    version = '0.1',
    license = "None",
    description = 'ACiM for KWS',
    author = "imec",
    author_email = "tom.vanderaa@imec.be",
    zip_safe = False,
    classifiers = CLASSIFIERS,
    keywords = "keyword spotting, machine learning, compute-in-memory",
    install_requires = ['numpy' ]
)
