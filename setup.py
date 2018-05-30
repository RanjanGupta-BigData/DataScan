#! python
from setuptools import setup, find_packages

setup(name='DataScan',
      version='0.1.0',
      author='Gerges Dib',
      author_email='',
      packages=['DataScan'],
      entry_points={'console_scripts': ['DataScan=DataScan.cli:cli']},
      install_requires=['numpy', 'pandas', 'xarray'],
      classifiers=['Programming Language :: Python :: 3.6']
      )

