<p align="center">
    <img src="https://raw.githubusercontent.com/DavidT3/DAXA/master/daxa/files/daxa-high-resolution-logo-black-on-white-background.png" width="500">
</p>

[![Documentation Status](https://readthedocs.org/projects/daxa/badge/?version=latest)](https://daxa.readthedocs.io/en/latest/?badge=latest)

# What is Democratising Archival X-ray Astronomy (DAXA)?

DAXA is a Python module designed to make the acquisition and processing of archives of X-ray astronomy data as
painless as possible. It provides a consistent interface to the downloading and cleaning processes of each telescope, 
allowing the user to easily create multi-mission X-ray archives, allowing for the community to make better use of
archival X-ray data. This process can be as simple or as in-depth as the user requires; if the default settings are 
used then data can be acquired and processed into an archive in only a few lines of code.

As the missions (i.e. telescopes) that should be included in the archive are defined, the user can filter the desired
observations based on a unique identifier (i.e. observation ID), on whether observations are near to a coordinate (or 
set of coordinates), and the time frame in which the observations were taken. As such it is possible to very quickly
identify what archival data might be available for a set of objects you wish to study. It is also possible to place
no filters on the desired observations, and as such process every observation available for a set of missions. 

Documentation is available **_[Under Development]_** on ReadTheDocs, and can be [found here](https://daxa.readthedocs.io), or
accessed by clicking on the documentation build status at the top of the README. The source for the documentation can
be found in the 'docs' directory in this repository.

# Which missions are supported?

**_DAXA is still in an early stage of development, and as such the list of supported telescopes is currently
limited. Support for more telescopes is either currently under development or being actively planned._** 

* XMM-Newton Pointed
* **_[Under Development]_** eROSITA Commissioning
* **_[Under Development - data acquisition implemented]_** NuSTAR

**_Chandra support is being planned as a priority_** - if you would like to help with this, please get in contact!

# Required telescope-specific software

DAXA makes significant use of existing processing software released by the telescope teams, and as such there are some
specific non-Python dependencies that need to be installed if that mission is to be included in a DAXA generated archive.

## An alternative to installing the dependencies yourself

**_[Under Development]_** - A docker image containing relevant telescope-specific software is being created. The 
built image will be released on DockerHub (or some other convenient platform), and the actual dockerfile used for
building the image will also be released for anyone to use/modify. The dockerfile is heavily inspired by/based off of 
the HEASoft docker image.

## XMM-Newton
Science Analysis System (SAS) - v14 or higher

## 

# Analysing the processed archives
Once an archive of cleaned X-ray data has been created, it can be analysed in all the standard ways, however you may
also wish to consider [X-ray: Generate and Analyse (XGA)](https://github.com/DavidT3/XGA), a companion module to DAXA.

XGA is also completely open source, and is a generalised tool for the analysis of X-ray emission from astrophysical 
sources. The software operates on a 'source based' paradigm, where the user declares sources or samples of objects
which are analogous to astrophysical sources in the sky, with XGA determining which data (if any) are relevant to a 
particular source, and providing a powerful (but easy to use) interface for the generation and analysis of data 
products. The module is fully documented, with tutorials and API documentation available (**support for telescopes 
other than XMM is still under development**).

# Problems and Questions
If you encounter a bug, or would like to make a feature request, please use the GitHub
[issues](https://github.com/DavidT3/DAXA/issues) page, it really helps to keep track of everything.

However, if you have further questions, or just want to make doubly sure I notice the issue, feel free to send
me an email at turne540@msu.edu





