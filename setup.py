from setuptools import find_packages
from setuptools import setup

setup(name='ctu',
      version='0.0.1',
      license='MIT',
      install_requires=[
          "numpy==1.16.4",
          "future==0.17.1",
          "dominate==2.3.5",
          "dill==0.3.0"
      ],
      extras_require={
          "pytorch": ["torch==1.1.0"],
          "torchvision": ["torchvision==0.3.0"],
          "scikit-image": ["scikit-image==0.15.0"],
          "tensorboard": ["tensorboard==1.14.0"],
          "tensorflow": ["tensorflow==1.14.0"]
      },
      packages=find_packages())
