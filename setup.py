from setuptools import setup

setup(
    name='atom3',
    packages=['atom3'],
    url='https://github.com/raphtown/atom3',
    version='0.1.4',
    description='3D Atomic Data Processing',
    long_description=open("README.rst").read(),
    author='Raphael Townshend',
    license='MIT',
    install_requires=[
        'biopython',
        'click',
        'easy-parallel',
        'h5py',
        'pandas',
        'scipy',
    ],
    entry_points={
        'console_scripts': [
            "atom3 = atom3.main:main",
        ]
    }
)
