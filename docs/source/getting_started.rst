Getting started
===============

Overview
^^^^^^^^

pyQSC is a python package for generating quasisymmetric stellarator configurations
using an expansion about the magnetic axis.
The underlying equations are explained in detail in
[LandremanSenguptaPlunk]_ and [LandremanSengupta]_.
PyQSC is closely related to the `fortran package here <https://github.com/landreman/quasisymmetry>`_,
except that pyQSC is written in pure python. This makes pyQSC user-friendly,
with no need for compilation, though it is slower.


Installation
^^^^^^^^^^^^

The recommended way to install pyQSC is to install it from `PyPI <https://pypi.org/project/qsc/>`_ using ``pip``::

    pip install qsc

If you prefer to see or edit the source code, you can first clone the repository using::

    git clone https://github.com/landreman/pyQSC.git

Then install the package to your local python environment with::

  cd pyQSC
  pip install -e .

The ``-e`` in the last command is optional.

.. [LandremanSenguptaPlunk] Landreman, Sengupta, and Plunk, *Journal of Plasma Physics* **85**, 905850103 (2019).
.. [LandremanSengupta] Landreman and Sengupta, *Journal of Plasma Physics* **85**, 815850601 (2019).
