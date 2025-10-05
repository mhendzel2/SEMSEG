"""
Setup script for SEMSEG - 3D FIB-SEM Segmentation and Quantification Program
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = []
    for line in f:
        line = line.strip()
        # Skip empty lines and comments
        if line and not line.startswith('#'):
            # Handle the mclahe special case
            if line.startswith('mclahe @'):
                requirements.append('mclahe @ https://github.com/VincentStimper/mclahe/archive/numpy.zip')
            # Skip optional dependencies marked with comments
            elif '# ' not in line or not line.startswith('#'):
                requirements.append(line.split('#')[0].strip())

setup(
    name='SEMSEG',
    version='1.0.0',
    author='Manus AI',
    author_email='',
    description='3D FIB-SEM Segmentation and Quantification Program',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mhendzel2/SEMSEG',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'scikit-image>=0.18.0',
        'h5py>=3.6.0',
        'tifffile>=2021.11.2',
        'PyYAML>=6.0',
        'Pillow>=9.0.0',
        'zarr',
        's3fs',
        'matplotlib>=3.5.0',
    ],
    extras_require={
        'deep_learning': [
            'tensorflow>=2.8.0',
        ],
        'advanced_segmentation': [
            'PyMaxflow>=1.2.13',
        ],
        'all': [
            'tensorflow>=2.8.0',
            'PyMaxflow>=1.2.13',
        ]
    },
    entry_points={
        'console_scripts': [
            'semseg=SEMSEG.__main__:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
