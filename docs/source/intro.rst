Introduction to DAXA
====================

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

Which missions are supported?
-----------------------------

*DAXA is still in an early stage of development, and as such the list of supported telescopes is currently
limited. Support for more telescopes is either currently under development or being actively planned.*

    * XMM-Newton Pointed
    * [Under Development - data acquisition implemented] eROSITA Commissioning
    * [Under Development - data acquisition implemented] NuSTAR Pointed
    * [Under Development - data acquisition implemented] Chandra
    * [Under Development - RASS/pointed data acquisition implemented] ROSAT

*If you wish to help with implementation of Chandra, NuSTAR, or some other mission, please get in contact!*

Analysing the processed archives
--------------------------------

Once an archive of cleaned X-ray data has been created, it can be analysed in all the standard ways, however you may
also wish to consider [X-ray: Generate and Analyse (XGA)](https://github.com/DavidT3/XGA), a companion module to DAXA.

XGA is also completely open source, and is a generalised tool for the analysis of X-ray emission from astrophysical
sources. The software operates on a 'source based' paradigm, where the user declares sources or samples of objects
which are analogous to astrophysical sources in the sky, with XGA determining which data (if any) are relevant to a
particular source, and providing a powerful (but easy to use) interface for the generation and analysis of data
products. The module is fully documented, with tutorials and API documentation available.