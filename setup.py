#! python
from setuptools import setup, find_packages

setup(name='DataScan',
      version='0.1',
      description='A library to handle Ultrasound NDE data.',
      url='http://github.com/dibgerge/DataScan',
      author='Gerges Dib',
      packages=['DataScan'],
      install_requires=['scipy>=1',
                        'xarray>=0.10',
                        'numpy>=1.14'],
      zip_safe=False)
