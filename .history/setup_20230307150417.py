from distutils.command.install_data import install_data
from ensurepip import version
from re import U
from struct import pack
import setuptools
setuptools.setup(
    name = "HypResEvalPy",
    version="0.1",
    author = "jingmengzhiyue",
    author_email = "jingmengzhiyue@gmail.com",
    description = "A Python package for super-resolution of hyperspectral images",
    long_description = "This Python package can calculate CC, DD, ERGAS, PSNR, RSNR, SAM, SSIM, and UIQI between two hyperspectral images. It also allows for spatial and spectral downsampling of hyperspectral images to generate low spatial resolution hyperspectral images and low spectral resolution multispectral images.",
    long_description_content_type = "text/markdown",
    url = "https://github.com/jingmengzhiyue/HypResEvalPy",
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