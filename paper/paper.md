---
title: 'DAXA: Traversing the X-ray desert by Democratising Archival X-ray Astronomy'
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
  - name: Department of Physics and Astronomy, Michigan State University, East Lansing, Michigan, USA
    index: 1
  - name: Department of Physics and Astronomy, University of Sussex, Brighton, UK
    index: 2
date: 20 March 2024
bibliography: paper.bib
---

# Summary
We introduce a new, open-source, Python module for the acquisition and processing of archival data from multiple X-ray 
telescopes, Democratising Archival X-ray Astronomy (hereafter referred to as [Daxa]{.smallcaps}). The aim of 
[Daxa]{.smallcaps} is to provide a consistent, easy-to-use, Python interface to the disparate X-ray telescope data 
archives, and their processing packages. We provide this interface for the majority of X-ray telescopes launched 
within the last 30 years. This module will enable much greater access to X-ray data for non-specialists, while 
preserving low-level control of processing for X-ray experts. The package is useful for identifying relevant 
observations of a single object of interest, but it excels at creating and managing multi-mission datasets for 
serendipitous or targeted studies of large samples of X-ray emitting objects. Once relevant observations are 
identified, the raw data can be downloaded (and optionally processed) through [Daxa]{.smallcaps}, or pre-processed 
event lists, images, and exposure maps can be downloaded if they are available. As we may enter an `X-ray desert', with 
no new X-ray missions coming online, during the next decade, archival data is going to take on an even greater 
importance than it already has, and enhanced access to those archives will be vital to the continuation of X-ray
astronomy.

# Statement of need

X-ray observations provide a powerful view of some of the most extreme processes in the Universe, and have had a 
profound impact on our understanding of many types of astrophysical objects; from in-solar-system objects, to 
supernovae, to galaxies and galaxy clusters. As such, access to X-ray data should be made as simple as possible, 
both for X-ray experts and non-specialists whose research benefits from a high-energy view; organisations such as 
the European Space Agency (ESA) and the High Energy Astrophysics Science Archive Research Center (HEASARC) have gone 
to great lengths to enable this access, and our software builds on their success. Through [Daxa]{.smallcaps}, most 
X-ray observatory archives are accessible through a single unified interface available in a programming language 
that is ubiquitous in astronomy (Python); locally searching for data relevant to a particular sample gives us the 
opportunity to better record and share the exact search parameters, through a Jupyter notebook for instance. X-ray data 
can also be particularly intimidating to those astronomers who have not used it before, which acts as a barrier to 
entry, limiting the reach and scientific impact of X-ray telescopes; it is in our interest to maximise the
use of these data, both to support X-ray astronomy through the `X-ray desert', and to persuade funding bodies of the 
great need for X-ray telescopes. [Daxa]{.smallcaps} is particularly powerful in this regard, as it provides a 
normalised, simple, interface to different backend software packages, allowing for the easy processing of X-ray data 
to a scientifically useful state; this is in addition to the ability to download pre-processed data from many of the 
data archives.

Almost every sub-field of astronomy, astrophysics, and cosmology has benefited significantly from X-ray coverage over 
the last three decades; calibrating weak-lensing mis-centering for galaxy cluster studies [@miscen], identifying 
**...COME UP WITH MORE IMPORTANT X-RAY PAPERS**, even probing the irradiation of exoplanets [@xrayirrexo]. The 
current workhorse X-ray observatories (_XMM_-Newton and _Chandra_; other telescopes are 
online but are not as generally useful) are ageing however, with _Chandra_ in particular experiencing a decline in 
low-energy sensitivity that limits possible science cases; these missions cannot last forever. If we are to enter an 
`X-ray desert', where the astrophysics community has only limited access to new X-ray observations from specialised 
missions like _Swift_, _NuSTAR_, and _XRISM_, then archival data (and serendipitous studies) take on an even greater 
value than they already hold. [Daxa]{.smallcaps} is part of an ecosystem of open-source software designed around the 
concept of enabling serendipitous studies of X-ray emitting objects, and can download and prepare X-ray observations 
for use with tools like 'X-ray: Generate and Analyse' ([Xga]{.smallcaps}; @xga). X-ray observations are perfectly suited
for the kind of archival study facilitated by [Daxa]{.smallcaps} and [Xga]{.smallcaps}, as they generally record the
time, position, and energy of each individual photon impacting the detector; this means that we can create images, 
lightcurves, and spectra for any object within the field-of-view, even if it was not the target. With this software, we
can enable the maximum exploitation of existing X-ray archives, both to traverse the `X-ray desert' and to ensure that we 
are fully prepared for future X-ray telescopes such as _Athena_ and _Lynx_. Having easy access to the whole history of 
X-ray observations of an object can provide extra context as to its astrophysics, and comes at no extra cost.

Finally, [Daxa]{.smallcaps} can be used to further one of the tenets of open-source science, reproducibility. Its 
management features both allow the user to keep track of their dataset, but also to version control it. If more data
become available, or existing data need to be reprocessed, then the version of the dataset can be automatically 
updated. Research publications can thus reference an exact version of a dataset, which can be reproduced without
offering the whole dataset for download.

[^*]: turne540@msu.edu

# Features

[Daxa]{.smallcaps} contains two types of class, mission classes and the archive class. Mission classes directly 
represent a telescope, or in some cases a particular use of a telescope (for instance there are separate classes for 
pointed and survey observations taken by _ROSAT_, as the characteristics of the data are quite different), and exist
to provide a Python interface with the up-to-date telescope observation database. Such mission classes allow the user
to easily identify data relevant to their objects of interest (both individual and samples) with various filtering 
methods (it is also possible to download the entire archive of a telescope); these include filtering on spatial 
position (determining whether a coordinate of interest is within the field-of-view), filtering on the time of the 
observation (also filtering on whether a specific coordinate was observed at a specific time, for whole samples with 
different coordinates and times of interest), and filtering on specific observation identifiers (ObsIDs) if they 
are already known. Each mission class has some knowledge of the characteristics of the telescope it represents (such as
the field-of-view) to make observation filtering easier. The user can also select only a subset of instruments, if the
telescope has more than one, to exclude any that may not contribute to their analysis. 

Once as set of relevant observations have been identified, for either a single mission or a set of missions, a 
[Daxa]{.smallcaps} multi-mission data archive can be declared. This will automatically download the selected data from
the various telescope archives, if the user had not already triggered that, and proceeds to ingest and organise the 
data so that it can be managed (and if necessary, updated) through the [Daxa]{.smallcaps} interface. Within the
Python interface it is easy to see and explore exactly what data files are available and for which observations. We
have also implemented user-friendly, multi-threaded, data preparation and cleaning routines for some telescopes (_XMM_ 
and _eROSITA_ in particular); fine control of the parameters that control these processes is retained, but default 
behaviours can be used if the user is unfamiliar with the minutiae of X-ray data preparation. Another key benefit of
reducing data with [Daxa]{.smallcaps} is the easy access to software logs through our interface, in case of suspected
problems during the reduction processes. The module is also capable of safely handling processing failures, simply 
recording at which processing step the failure occurred for a particular ObsID. 

All of this information is retained permanently, not just while the initial [Daxa]{.smallcaps} processes are 
running. Any [Daxa]{.smallcaps} archive can be loaded back into memory after the fact, once again providing access 
to the stored log information. At this point the archives can also be updated, either by searching for new data from
the existing missions, adding data from a different mission, or re-processing specific observations to achieve more
scientifically useful data. Any such change will be recorded, and processed observations version controlled, so that 
the data archive can have a specific version that refers to its exact state at any given time; this version can be
referred to in published work using the data archive. Each data archive is also capable of creating a file that
other [Daxa]{.smallcaps} users can import, and which will recreate the data archive by downloading the same data, and
processing it in the same way; this renders making fully processed, and large, X-ray data files available with 
a piece of research is unnecessary.

# Existing software packages

There are no direct analogues to our module, though we must acknowledge the many pieces of software (and data
archives), without which it would have been much more difficult to create [Daxa]{.smallcaps}. Data access is made 
possible primarily by the HEASARC data archive, though the Astroquery [@astroquery] package is also used. 
HEASARC provides an online interface to query their data archive, which has similar functionality to some of the 
filtering methods of mission classes in [Daxa]{.smallcaps} (though we provide slightly more functionality in 
that regard), and they provide Python SQL examples to access the data, but none of the data management and cleaning 
functionality that we include. 

Also worthy of mention are the various telescope-specific software packages that underpin [Daxa]{.smallcaps}'s ability
to perform data preparation and cleaning. Particularly important are the _XMM_ Science Analysis System (SAS; @sas) and 
the complementary extended SAS (eSAS; @esascook) packages, which allow us to provide simple Python interfaces to the
complex, multi-step, processes that are required to prepare raw _XMM_ data for scientific use. The analogous 
_eROSITA_ Science Analysis Software System (eSASS; @erosita) must also be mentioned, as it provides the tools needed to
reduce and prepare _eROSITA_ data. In this vein we must also mention the HEASoft package, which is almost ubiquitous in
X-ray data analyses, and is used by both SAS and eSASS.

Another related software package is the other module in our open-source X-ray astronomy ecosystem, X-ray: Generate 
and Analyse ([Xga]{.smallcaps}; @xga). It has none of the same features as [Daxa]{.smallcaps}, as it exists to 
analyse large sets of X-ray data, we created [Daxa]{.smallcaps} to create and manage the kind of dataset required for
[Xga]{.smallcaps} to attain maximum usefulness.  

[Daxa]{.smallcaps} is much greater than the sum of its parts, but would not have been possible without the existing
software packages it builds upon; we hope that it only enhances the value that astrophysicists derive from the other
software we have mentioned.

# Research projects using DAXA

[Daxa]{.smallcaps}'s feature-set has reached a minimum viable product level, and has been useful for several research 
projects over different collaborations; we anticipate that the number of research projects benefiting from it will
increase significantly. The primary use to which [Daxa]{.smallcaps} has been put is to assemble the multi-mission 
X-ray dataset (_XMM_, _Chandra_, _eROSITA_, _Swift_, and _ROSAT_) for the X-ray follow-up component of the Local Volume
Complete Cluster Survey (LoVoCCS). It was used to identify the relevant observations, download, and process them, as 
well as to organise the significant number of files and make it easier for the dataset to be served to the X-ray 
community. Construction and administration of such large, complicated, multi-mission datasets is rendered quick and
easy.

The X-ray Cluster Science (XCS; formerly known as the _XMM_ Cluster Survey) collaboration now uses [Daxa]{.smallcaps} to
create and manage their processed X-ray archive; particularly useful is [Daxa]{.smallcaps}'s support for telescopes 
other than _XMM_, which has allowed the serendipitous science undertaken by XCS to expand to the use of different 
telescopes. These telescopes are complementary to _XMM_, and also increase the sky coverage, which in turn increases
the likelihood that an object of interest has an accessible X-ray observation.

As [Daxa]{.smallcaps} now supports XCS, it has contributed to a research project that has measured X-ray properties
(spectral, time-series, and photometric) for every LOFAR source that falls on an _XMM_ observation (a [Daxa]{.smallcaps} 
generated dataset was used for an [Xga]{.smallcaps} analysis). This kind of bulk analysis is trivial when our software
packages are utilised, and will result in a comprehensive catalogue that is invaluable to the radio astronomy community.

Finally, [Daxa]{.smallcaps} has been used to identify _XMM_ and _Chandra_ (alongside other telescopes, though they 
play only a supporting role) observations of a series of galaxy groups that appear in the foreground of UV bright
quasars. Absorption features that indicate the presence of Oxygen VI were identified in the spectra of several of 
the quasars, and the data that [Daxa]{.smallcaps} identified and retrieved allowed for an exploration of the hot-gas
properties of these groups.

# Future Work

The most significant new features implemented in [Daxa]{.smallcaps} will be new mission classes added when new X-ray 
telescope archives become available, or one of the existing missions that we have not yet implemented is added (for 
instance _XMM_ observations taken whilst slewing). We will also seek to include support for more telescope-specific 
cleaning methods taken from their backend software; additionally we wish to implement our own generic processing and 
cleaning techniques where possible, applicable to multiple missions. We also aim to include source detection 
capabilities; specifically techniques that are generally applicable to multiple missions whilst taking into account
instrument-specific effects. 

# Acknowledgements
DT and MD are grateful for support from the National Aeronautic and Space Administration Astrophysics Data Analysis 
Program (NASA80NSSC22K0476). 

KR and PG acknowledge support from the UK Science and Technology Facilities Council via grants ST/T000473/1 and ST/X001040/1.
JP acknowledges support from the UK Science and Technology Facilities Council via grants ST/X508822/1.

# References
