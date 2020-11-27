The objective of this package is to implement some shape feature extraction for functional data analysis (FDA).

Documentation
=============

FDA aims at studying dataset where each individual **x_i** is a realisation of an unknown function **f** which depend on a continuous variable **u**. 

As we work with multivariate data each  **x_i** is a vector
To approximate this variable we can use a linear combination of functions (e.g Bspline, Fourier).
The approximation by a known basis of function enable us to estimate its derivatives to capture shape features (e.g curvature,velocity,arc length).

This package is based on scikit-fda package, see notebooks for examples.

Installation
============
Currently, you only need to install the library scikit-fda https://fda.readthedocs.io/en/stable/ and its dependencies to use this package
Note: fda requires Visual studio Build tools as compiler

Installation 
------------------------

.. code:: bash

    git clone https://github.com/Guillaume-Bernard/curve_shape_analysis.git
    pip install ./curve_shape_analysis

References
============
- https://www.sciencedirect.com/science/article/abs/pii/S0950705120302835 : Clément Lejeune, Josiane Mothe, Adil Soubki, Olivier Teste. Shape-based outlier detection in multivariate functional data. Knowledge-based Systems. 2020.
- https://openproceedings.org/2020/conf/edbt/paper_236.pdf : Clément Lejeune, Josiane Mothe, Olivier Teste. Outlier detection in multivariate functional data based on a geometric aggregation. Proceedings of EDBT conference 2020.
- Functional Data Analysis with R and MATLAB 2009 Authors: Ramsay, James O., Hooker, Giles, Graves, Spencer
- Scikit-fda: https://fda.readthedocs.io/en/stable/

Contributions
=============

This package is based on `Clément Lejeune's <https://dblp.uni-trier.de/pid/261/2070.html>`_ paper : Shape-based outlier detection in multivariate functional data


The people involved in the development are Guillaume BERNARD, Clément LEJEUNE, Sandra FERRIERES and Olivier TESTE
