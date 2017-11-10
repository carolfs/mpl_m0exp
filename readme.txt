Copyright 2017 Carolina Feher da Silva <carolfsu@gmail.com>

This is the data and code for the paper:

Feher da Silva, C.; Victorino, C. G.; Caticha, N.; Baldo, M. V. C.
Exploration and recency as the main proximate causes of probability
matching: a reinforcement learning analysis
Scientific Reports 7, Article number: 15326 (2017)
doi:10.1038/s41598-017-15587-z

The behavioral data is saved in CSV format as bdata.csv

There are three MPL model implementations, in Python 3, Stan, and C++ (as a
Python 3 module):
MPL in Stan: model-mpl.stan
MPL in Python: mpl_python.py
MPL in C++ as a Python module: mpl.cpp

To compile the MPL Python module, run:
$ python setup.py build
and copy the module to the project's root directory.
