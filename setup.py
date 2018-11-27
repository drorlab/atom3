from setuptools import setup

setup(
    name='atom_data',
    packages=['atom_data'],
    url='https://github.com/raphtown/atom_data',
    version='0.1.2',
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
