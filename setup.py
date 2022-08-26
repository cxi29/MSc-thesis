from setuptools import setup
from setuptools import find_packages

setup(name='DropConnect_LDPC',
      version='0.1.0',
      description='LDPC Dropconnect mask',
      author='Xi Chen',
      author_email='puddincham@gmail.com',
      license='MIT',
      install_requires=['keras'],
      packages=find_packages())