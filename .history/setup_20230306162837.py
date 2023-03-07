from distutils.command.install_data import install_data
from ensurepip import version
from re import U
from struct import pack
import setuptools
setuptools.setup(
    name = "pyhsr",
    version="0.1",
    author = "jingmengzhiyue",
    author_email = "jingmengzhiyue@gmail.com",
    description = "A Python package for super-resolution of hyperspectral images",
    long_description = "A Python package for super-resolution of hyperspectral images",
    long_description_content_type = "text/markdown",
    url = "https://github.com/jingmengzhiyue/pyhsr",
    packages =  setuptools.find_packages(),
    classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    ],
    install_requires = [
    'numpy',
    ],
    python_requires = ">=3",

)