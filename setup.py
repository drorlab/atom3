from setuptools import find_packages, setup

setup(
    name='atom-data',
    packages=find_packages(),
    url='https://github.com/raphtown/atom-data',
    version='0.1.0',
    description='Atomic Data Processing',
    long_description=open("README.rst").read(),
    author='Raphael Townshend',
    license='MIT',
    install_requires=[
        'biopython',
        'easy-parallel',
        'h5py',
        'pandas',
        'scipy',
    ],
)
