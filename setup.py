#!/usr/bin/env python

from setuptools import setup

setup(
    name="timedelay",
    license="GNU",
    packages=["timedelay"],
    install_requires=['numpy>=1.10','astropy>=1.0','seaborn', 'tqdm','']
)