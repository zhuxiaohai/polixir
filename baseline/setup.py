import sys
from setuptools import setup, find_packages

assert sys.version_info.major == 3, \
    "This repo is designed to work with Python 3." \
    + "Please install it before proceeding."

assert sys.version_info.minor in [6, 7, 8], \
    "This repo uses revive sdk which only supports Python 3.6 ~ 3.8 for now."

setup(
    name='offlinerl_baseline',
    author='Polixir Technologies Co., Ltd.',
    packages=find_packages(),
    version="1.0.0",
    install_requires=[
        'numpy',
        'pandas',
        'torch>=1.8',
        'gym',
        'stable-baselines3',
        'ipykernel',
        'notebook'
    ],
    url="https://codalab.lisn.upsaclay.fr/competitions/823"
)
