from setuptools import setup, find_packages
#import os
#os.chdir(os.pardir)

setup(
    name='TDTR_fitting',
    version='0.161',    
    description='fitting code for thermoreflectance experiments',
    url='https://github.com/ExSiTE-Lab/TDTR_fitting',
    author='Thomas W. Pfeifer',
    author_email='twp4fg@virginia.edu',
    packages=["TDTR_fitting"],
    install_requires=['numpy']
)