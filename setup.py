from setuptools import setup, find_packages
from setuptools.extension import Extension


setup(name='curve_shape_analysis',
      url='https://github.com/Guillaume-Bernard/curve_shape_analysis',
      maintainer='Guillaume BERNARD',
      maintainer_email='bernardguiugi@hotmail.fr',
      include_package_data=True,
      packages=find_packages(),
      python_requires='>=3.6, <4',
      install_requires=[
          'cython',
          'dcor',
          'findiff',
          'matplotlib',
          'mpldatacursor',
          'multimethod>=1.2',
          'numpy>=1.16',
          'pandas',
          'rdata',
          'scikit-datasets[cran]>=0.1.24',
          'scikit-learn>=0.20',
          'scipy>=1.3.0',
          'POT',
          'pymanopt',
          'autograd',
          'scikit-fda'
      ])
