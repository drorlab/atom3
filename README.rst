atom3
=====

Processing of 3D atomic data.

Installation
------------

.. code-block:: console

    $ pip install atom3 

Usage
-----

Process a PDB file to a pickled pandas dataframe.

.. code-block:: console

    $ atom3 parse sample/11as.pdb1.gz sample/parsed

Derive subunits of a parsed dataframe.

.. code-block:: console

    $ atom3 complex sample/parsed sample/complexes.dill

Split subunits into interacting pairs.

.. code-block:: console

    $ atom3 pairs sample/complexes.dill sample/pairs

For help on commands.

.. code-block:: console

    $ atom3 -h
    $ atom3 parse -h
