"""
TorchLight Setup Script
=======================

This is the installation script for the `torchlight` package, a minimal
framework built on PyTorch for streamlined deep learning workflows.

Purpose:
    Defines package metadata and dependencies for distribution via pip.

Usage:
    Install in development mode:
        pip install -e .
    
    Build distribution packages:
        python setup.py sdist bdist_wheel
    
    Upload to PyPI:
        twine upload dist/*

Package Structure:
    - Uses `find_packages()` to automatically discover all Python packages
    - Currently has no external dependencies (`install_requires=[]`)
    - Follows standard Python packaging conventions

Version: 1.0
Author: [Your Name/Organization]
License: [Specify License, e.g., MIT, Apache 2.0]
Project URL: [GitHub/GitLab repository URL]
"""

from setuptools import find_packages, setup

setup(
    name='torchlight',
    version='1.0',
    description='A mini framework for pytorch',
    packages=find_packages(),
    install_requires=[])
