---
title: 'DAXA: Democratising archival X-ray astronomy through the easy creation of multi-mission datasets'
tags:
  - Python
  - Astronomy
  - Astrophysics
  - X-ray astronomy
  - Galaxy clusters
  - AGN
  - Instrumentation
  - XMM
  - eROSITA
  - Chandra
  - NuSTAR
  - Swift
  - ROSAT
  - ASCA
  - Suzaku
authors:
  - name: David J. Turner[^*]
    orcid: 0000-0001-9658-1396
    affiliation: 1,2
  - name: Jessica E. Pilling
    orcid: 0000-0002-3211-928X
    affiliation: 2
  - name: Megan Donahue
    orcid: 0000-0002-2808-0853
    affiliation: 1
  - name: Paul A. Giles
    orcid: 0000-0003-4937-8453
    affiliation: 2
  - name: Kathy Romer
    orcid: 0000-0002-9328-879X
    affiliation: 2
  - name: Agrim Gupta
    affiliation: 1
  - name: Toby Wallage
    affiliation: 2
affiliations:
  - name: Department of Physics and Astronomy, Michigan State University, East Lansing, Michigan, 48824, USA
    index: 1
  - name: Department of Physics and Astronomy, University of Sussex, Brighton, BN1 9QH, UK
    index: 2
date: 12 March 2024
bibliography: paper.bib
---

# Summary
We introduce a new, open-source, Python module for the acquisition and processing of archival data from multiple X-ray 
telescopes, Democratising Archival X-ray Astronomy (hereafter referred to as [Daxa]{.smallcaps}). The aim of 
[Daxa]{.smallcaps} is to provide a consistent, easy-to-use, Python interface with the disparate X-ray telescope data 
archives, and their processing packages. We provide this interface for the majority of X-ray telescopes launched 
within the last 30 years. This module will enable much greater access to X-ray data for non-specialists, while 
preserving low-level control of processing for X-ray experts. The package is useful for identifying relevant 
observations of a single object of interest, but it excels at creating and managing multi-mission datasets for 
serendipitous studies of large samples of X-ray emitting objects. Once relevant observations are identified, the raw 
data can be downloaded (and optionally processed) through [Daxa]{.smallcaps}, or pre-processed event lists, images, and 
exposure maps can be downloaded if they are available. As we may enter an 'X-ray desert', with no new 
missions coming online, within the next decade, archival data is going to take on an even greater importance than
it already has, and easy access to those archives will be vital to the continuation of X-ray astronomy.

# Statement of need

The study of X-ray emission from astrophysical objects provides a powerful view of some of the most extreme processes 
in the Universe, has had a profound impact on our understanding of many types of objects; from in-solar-system 
objects, to supernovae, to galaxies and galaxy clusters. As such, access to X-ray data should be made as simple as possible, 
both for X-ray experts and to those non-specialists whose research could benefit from examining their sources at 
higher energies; organisations such as the European Space Agency (ESA) and the High Energy Astrophysics Science Archive 
Research Center (HEASARC) have gone to great lengths to enable this access, and we have built on their success to 
create our software. Through [Daxa]{.smallcaps}, all major X-ray observatory observation archives are accessible 
through a single unified interface available in a programming language that is almost ubiquitous in astronomy 
(Python), and can be searched in a variety of ways to find only the data relevant to objects in a particular 
study. X-ray data can be particularly intimidating to those astronomers who have not used it before, which acts
as a barrier to entry, limiting the reach and scientific of X-ray telescopes, things we should be striving to 
maximise. Our software is particularly powerful in this regard, as it provides a normalised and simple interface to 
different backend software packages, allowing for the easy processing of X-ray data to a scientifically useful 
state; this is in addition to the ability to download pre-processed data from many of the data archives.

Almost every sub-field of astronomy, astrophysics, and cosmology has 
benefited significantly from X-ray coverage over the last three decades. The current workhorse X-ray observatories 
(_XMM_-Newton and _Chandra_; other telescopes are online but are not as generally useful) are ageing however, with 
_Chandra_ in particular experiencing a decline in low-energy sensitivity that limits possible science cases; these 
missions cannot last forever. 

Testing [@xga]

[^*]: turne540@msu.edu

# Features


# Existing software packages
There is no exact analogue to 

# Research projects using DAXA [Daxa]{.smallcaps}

[Daxa]{.smallcaps}

# Future Work


# Acknowledgements
DT and MD are grateful for support from the National Aeronautic and Space Administration Astrophysics Data Analysis 
Program (NASA80NSSC22K0476). KR and PG acknowledge support from the UK Science and Technology Facilities Council via 
grants ST/T000473/1 and ST/X001040/1.

ADD JESS' STFC CODE

DO I INCLUDE THIS?? We acknowledge contributions to the _XMM_ Cluster Survey from A. Bermeo, M. Hilton, P. J. Rooney, 
S. Bhargava, L. Ebrahimpour, R. G. Mann, M. Manolopoulou, J. Mayers, E. W. Upsdell, C. Vergara, P. T. P. Viana, 
R. Wilkinson, C. A. Collins, R. C. Nichol, J. P. Stott, and others.

# References
