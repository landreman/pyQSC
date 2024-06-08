===================
pyQSC Documentation
===================

``pyQSC`` is a python package for generating quasisymmetric `stellarator
<https://en.wikipedia.org/wiki/Stellarator>`_ configurations using the
near-axis expansion.
The source code can be found at https://github.com/landreman/pyQSC.
The following components are included:

- Creation of first and second order quasisymmetric stellarators.
- Plot resulting surfaces, magnetic field properties and magnetic axis
- Output to VMEC
- Calculate the grad B and grad grad B tensors
- For second order configurations, calculate the Mercier criterion
- Get_dofs and set_dofs functions allowing it to be used with SIMSOPT

``pyQSC`` is fully open-source, and anyone is welcome to
make suggestions, contribute, and use.

.. toctree::
   :maxdepth: 3

   getting_started
   usage
   outputs
   api
   source
