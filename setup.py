"""Temporal Convolutional Neural Networks Setup
"""

import os
import setuptools
import codecs

import tconvnet

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with codecs.open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='tconvnet',
    version=tconvnet.__version__,
    description='Temporal Convolutional Neural Networks',
    long_description=long_description,
    url='https://github.com/dicarlolab/tconvnet',
    author='DiCarlo Lab',
    author_email='qbilius@mit.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7'
    ],
    packages=['tconvnet'],
    # packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['numpy', 'networkx', 'tensorflow>=1.0.0rc0'],
    extras_require={
        'test': ['nose'],
    }
)