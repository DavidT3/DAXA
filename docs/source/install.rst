Installing and Configuring DAXA
==============

This is a slightly more complex installation than many Python modules, but shouldn't be too difficult. If you're
having issues feel free to contact me.

The Module
----------

**DAXA is not yet available on PyPi or Conda, though it will be in the future**

I strongly recommend that you make use of Python virtual environments, or (even better) Conda/Mamba virtual environments when installing DAXA.

You can fetch the current working version from the git repository, and install it using the setup.py

.. code-block::

    git clone https://github.com/DavidT3/DAXA
    cd DAXA
    python setup.py install

Alternatively you could use the 'develop' option so that any changes you pull from the remote repository are reflected without having to reinstall DAXA.

.. code-block::

    git clone https://github.com/DavidT3/DAXA
    cd DAXA
    python setup.py develop

Once installed, you can import DAXA in the usual way (the command will be lowercase, 'import daxa'). If you have stayed
in the DAXA directory cloned from GitHub and opened Python there then it is possible to 'import DAXA', but that will behave
very strangely as it hasn't actually imported the module, but the directory.

Required Dependencies
---------------------

* XMM
    - Science Analysis System (SAS) - v14 or above
    - HEASoft (lcurve is required for XMM processing) - tested on v6.29 and v6.31

All required Python modules can be found in requirements.txt, and should be added to your system during the installation of DAXA.

Excellent installation guides for `SAS <https://www.cosmos.esa.int/web/xmm-newton/sas-installation>`_ and
`HEASoft <https://heasarc.gsfc.nasa.gov/lheasoft/install.html>`_ exist.

A docker container for DAXA (and the related module XGA) is in development, where all dependencies will already be installed.


Configuring DAXA
---------------

The first run of DAXA will create a configuration file, which by default will be ~/.config/daxa/daxa.cfg. This does not **need** to be configured
by the user, but you can use it to set the default directory for saving DAXA outputs, as well as an override of the number of cores that DAXA is allowed to use.
