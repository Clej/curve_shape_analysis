The objective of this package is to implement some features for functional data analysis (FDA).

Documentation
=============

|FDA aims at studying dataset where each individual **x_i** is a realisation of an unknown function **f** which depend on a continuous variable **u**. 
This continuous variable can be univariate (1 variable) or multivariate (2 or more variables).
|To approximate this variable we can use a linear combination of functions (e.g Bspline, Fourier).
The approximation by a known basis of function enable us to estimate its derivatives.

Installation
============
Currently, you only need to install the library scikit-fda https://fda.readthedocs.io/en/stable/ and its dependencies to use this package

Installation 
------------------------

.. code:: bash

    git clone https://github.com/Guillaume-Bernard/scikit_fda_shape_analysis.git
    pip install ./scikit_fda_shape_analysis

Reference
============
Clement Lejeune paper : https://www.sciencedirect.com/science/article/abs/pii/S0950705120302835

Functional Data Analysis with R and MATLAB 2009 Authors: Ramsay, James O., Hooker, Giles, Graves, Spencer

Scikit-fda: https://fda.readthedocs.io/en/stable/

Contributions
=============

The people involved in the development are Guillaume BERNARD, Clement LEJEUNE and Olivier TESTE
