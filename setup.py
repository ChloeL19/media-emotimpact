#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 19:48:45 2018

@author: chloeloughridge
"""

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['keras==2.1.6', 'h5py'] #TODO CONFIRM

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
    
)