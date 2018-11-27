atom_data
=========

Utilities for atomic data processing.

Installation
------------

.. code-block:: console

    $ pip install atom_data 

Usage
-----

Process a PDB file to a pickled pandas dataframe.

.. code-block:: console

    $ python main.py parse sample/11as.pdb1.gz sample/parsed

Derive subunits of a parsed dataframe.

.. code-block:: console

    $ python main.py complex sample/parsed sample/complexes.dill

Split subunits into interacting pairs.

.. code-block:: console

    $ python main.py pairs sample/complexes.dill sample/pairs

For help on commands.

.. code-block:: console

    $ python main.py -h
    $ python main.py pairs -h
