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

## Required Dependencies

XMM SAS v14 or above (I'll make this prettier later on)

# Problems and Questions
If you encounter a bug, or would like to make a feature request, please use the GitHub
[issues](https://github.com/DavidT3/DAXA/issues) page, it really helps to keep track of everything.

However, if you have further questions, or just want to make doubly sure I notice the issue, feel free to send
me an email at turne540@msu.edu





