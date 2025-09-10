Installing and Configuring DAXA
==============

This can be a slightly more complex installation than many Python modules, but shouldn't be too difficult. If you're
having issues feel free to contact us.

The Module
----------

We **strongly recommend** that you make use of Python virtual environments, or (even better) Conda/Mamba virtual environments when installing DAXA.

DAXA is available on the popular Python Package Index (PyPI), and can be installed like this:

.. code-block::

    pip install daxa

You can also fetch the current working version from the git repository, and install it (this method has replaced 'python setup.py install'):

.. code-block::

    git clone https://github.com/DavidT3/DAXA
    cd DAXA
    python -m pip install .

Alternatively you could use the 'editable' option (this has replaced running setup.py and passing 'develop') so that any changes you pull from the remote repository are reflected without having to reinstall DAXA.

.. code-block::

    git clone https://github.com/DavidT3/DAXA
    cd DAXA
    python -m pip install --editable .

We also provide a Conda lock file in the conda_envs directory (see `conda-lock GitHub README <https://github.com/conda/conda-lock/README.md>`_ on how to install conda-lock), which can be used to create an Anaconda environment with the required dependencies:

.. code-block::
    conda-lock install -n <YOUR ENVIRONMENT NAME GOES HERE>
    conda activate <YOUR ENVIRONMENT NAME GOES HERE>

Once installed, you can import DAXA in the usual way (the command will be lowercase, 'import daxa'). If you have stayed
in the DAXA directory cloned from GitHub and opened Python there then it is possible to 'import DAXA', but that will behave
very strangely as it hasn't actually imported the module, but the directory.

Required Dependencies for Processing Data
-----------------------------------------

* XMM-Newton
    - Science Analysis System (SAS) - v20 or above
    - HEASoft (lcurve is required for XMM processing) - tested on v6.29 and v6.31

* Chandra
    - Chandra Interactive Analysis of Observations (CIAO) - v4.16 or above (installable through Conda)

* eROSITA
    - eROSITA Science Analysis Software System (eSASS) - both eSASS4EDR and eSASS4DR1 should be supported, but it is up to the user to choose [which is suitable for their use case](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/)
    - HEASoft - tested on v6.29 and v6.31


All required Python modules can be found in requirements.txt, and should be added to your system during the installation of DAXA.

Excellent installation guides for `SAS <https://www.cosmos.esa.int/web/xmm-newton/sas-installation>`_ and
`HEASoft <https://heasarc.gsfc.nasa.gov/lheasoft/install.html>`_ exist.


Configuring DAXA
---------------

The first run of DAXA will create a configuration file, which by default will be ~/.config/daxa/daxa.cfg. This does not **need** to be configured
by the user, but you can use it to set the default directory for saving DAXA outputs, as well as an override of the number of cores that DAXA is allowed to use.
