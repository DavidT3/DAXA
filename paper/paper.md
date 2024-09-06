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
  - name: Ray Wang
    orcid: 0000-0003-2102-8646
    affiliation: 1
affiliations:
  - name: Department of Physics and Astronomy,  Michigan State University,  East Lansing, Michigan, USA
    index: 1
  - name: Department of Physics and Astronomy, University of Sussex, Brighton, UK
    index: 2
date: 29 August 2024
bibliography: paper.bib
---

# Summary
We introduce a new, open-source, Python module for the acquisition and processing of archival data from many X-ray 
telescopes, Democratising Archival X-ray Astronomy (hereafter referred to as [Daxa]{.smallcaps}). The aim of 
[Daxa]{.smallcaps} is to provide a unified, easy-to-use, Python interface to the disparate X-ray telescope data 
archives and their processing tools. We provide this interface for the majority of X-ray telescopes launched 
within the last 30 years. This module enables much greater access to X-ray data for non-specialists, while 
preserving low-level control of processing for X-ray experts. It is useful for identifying relevant 
observations of a single object of interest, but it excels at creating and managing multi-mission datasets for 
serendipitous or targeted studies of large samples of X-ray emitting objects. Once relevant observations are 
identified, the raw data can be downloaded (and optionally processed) through [Daxa]{.smallcaps}, or pre-processed 
event lists, images, and exposure maps can be downloaded if they are available. With at least a decade-long `X-ray 
desert' on the horizon, archival data will take on even greater importance, and enhanced access to those archives 
will be vital to the continuation of X-ray astronomy.

# Statement of need
X-ray observations provide a powerful view of some of the most extreme processes in the Universe, and have had a 
profound impact on our understanding of many types of astrophysical objects. As such, access to X-ray data should be 
made as simple as possible, both for X-ray experts and non-specialists whose research benefits from a high-energy 
view; organisations such as the European Space Agency (ESA) and the High Energy Astrophysics Science Archive Research 
Center (HEASARC) have gone to great lengths to enable this access, and our software builds on their success. 
Through [Daxa]{.smallcaps}, most X-ray observatory archives are accessible through a single unified interface 
available in a programming language that is ubiquitous in astronomy (Python); locally searching for data relevant 
to a particular sample gives us the opportunity to better record and share the exact search parameters, through a 
Jupyter notebook for instance. X-ray data can also be particularly intimidating to those astronomers who have not 
used it before, which acts as a barrier to entry, undermining the open-source nature of X-ray astronomy data, and 
limiting the reach and scientific impact of X-ray telescopes; it is in our interest to maximise the use of these 
data, both to support X-ray astronomy through the `X-ray desert', and to persuade funding bodies of the great need for 
further X-ray telescopes. Our solution is to provide a normalised interface to different backend software 
packages (some of which can be difficult for new users), allowing for the easy processing of X-ray data to a 
scientifically useful state; this is in addition to the ability to download pre-processed data from many of the 
data archives.

![A flowchart giving a brief overview of the [Daxa]{.smallcaps} workflow. We indicate the different ways that [Daxa]{.smallcaps} can be used to access, process, and use archival X-ray data. \label{fig:flowchart}](figures/daxa_paper_flowchart.pdf)

Every sub-field of astronomy, astrophysics, and cosmology has benefited significantly from X-ray coverage over 
the last three decades; the observation of X-ray cavities in galaxy clusters caused by central AGN helped to shed light 
on the cooling-flow problem [@cavities]; further X-ray observations allowed for the measurement of spatially-resolved 
entropy in hundreds of clusters, dramatically increasing understanding of cooling and heating processes in their 
cores; quasi-periodic eruptions (QPE) from active galactic nuclei (AGN) were discovered [@qpedisco]; the high-energy 
view of young stars gave insights into their magnetic fields and stellar winds [@coup; @xest]; calibrating 
mis-centering for galaxy cluster weak-lensing studies helped constrain cosmological parameters [@miscen]; and X-rays 
even helped probe the irradiation of exoplanets [@xrayirrexo]. Indeed, X-ray telescopes have created many 
entirely new fields of study; they provided the first evidence of X-ray sources outside the solar 
system [@theOG]; discovered the first widely accepted black hole, and launched the study of supernova 
remnants [@cygx1andfriends]; and found ionized, volume-filling, gas within the Coma galaxy cluster (the intra-cluster 
medium) [@clusterdisco], with the implication that clusters were more than collections of galaxies. These 
non-exhaustive lists make evident the importance of X-ray observations to the astronomy, astrophysics, and cosmology 
communities.

The current workhorse X-ray observatories (_XMM_[@xmm] and _Chandra_; other telescopes are 
online but are more specialised) are ageing, with _Chandra_ in particular experiencing a decline in 
low-energy sensitivity that might limit science cases; these missions cannot last forever. If we are to enter an 
X-ray desert, where the astrophysics community has only limited access to new X-ray observations from specialised 
missions like _Swift_ [@swift], _NuSTAR_ [@nustar], and _XRISM_ [@xrism], then archival data (and serendipitous studies) 
take on an even greater value than they already hold. [Daxa]{.smallcaps} is part of an ecosystem of open-source software 
designed around the concept of enabling serendipitous studies of X-ray emitting objects, and can download and prepare 
X-ray observations for use with tools like 'X-ray: Generate and Analyse' ([Xga]{.smallcaps}; @xga). X-ray observations 
are uniquely well suited for the kind of archival study facilitated by [Daxa]{.smallcaps} and [Xga]{.smallcaps}, as 
they generally record the time, position, and energy of each individual photon impacting the detector (true for all 
missions currently implemented in [Daxa]{.smallcaps}); this means that we can create images, lightcurves, and spectra 
for any object detected within the field-of-view, even if it was not the target. With this software, we
enable the maximum use of existing X-ray archives, to traverse the X-ray desert and ensure that we 
are fully prepared for future X-ray telescopes such as _Athena_ [@athena] and _Lynx_ [@lynx]. Having easy access to 
the whole X-ray observation history of an object can provide valuable astrophysical context at little extra cost.

[^*]: turne540@msu.edu

# Features

[Daxa]{.smallcaps} contains two types of Python class: mission classes and the archive class. Mission classes directly 
represent a telescope or survey (for instance there are separate classes for pointed and survey observations taken by 
_ROSAT_ [@rosat], as the characteristics of the data are quite different), and exist to provide a Python interface 
with the current telescope observation database. Such mission classes allow the user to easily identify data relevant 
to their objects of interest with various filtering methods (it is also possible to download the entire archive of 
a telescope); these include filtering on spatial position (determining whether a coordinate of interest is within the 
field-of-view), filtering on the time of the observation (also filtering on whether a specific coordinate was observed 
at a specific time, for whole samples with different coordinates and times of interest), and filtering on specific 
observation identifiers (ObsIDs) if they are already known. Each mission class has some knowledge of the 
characteristics of the telescope it represents (such as the field-of-view) to make observation filtering easier. The 
user can also select a subset of instruments, if the telescope has more than one, to exclude any that may not 
contribute to their analysis. 

Once a set of relevant observations have been identified, for either a single mission or a set of missions, a 
[Daxa]{.smallcaps} data archive can be declared. This will automatically download the selected data from
the various telescope archives, and proceeds to ingest and organise the data so that it can be managed (and if 
necessary, updated) through the [Daxa]{.smallcaps} interface. We have also implemented user-friendly, multi-threaded, 
data preparation and cleaning routines for some telescopes (_XMM_ and _eROSITA_ in particular, though more will be 
added); fine control of the parameters that configure these processes is retained, but default 
behaviours can be used if the user is unfamiliar with the minutiae of X-ray data preparation. Another key benefit of
reducing data with [Daxa]{.smallcaps} is the easy access to data logs through its interface, in case of 
suspected problems during the reduction processes. The module is also capable of safely handling processing 
failures, recording at which processing step the failure occurred for a particular ObsID. 

All of this information is retained permanently, not just while the initial [Daxa]{.smallcaps} processes are 
running. Any [Daxa]{.smallcaps} archive can be loaded back in after the initial processing, once again providing access 
to the stored logs, and processing information. At this point the archives can also be updated, either by searching 
for new data from the existing missions, adding data from a different mission, or re-processing specific observations 
to achieve more scientifically useful data. Each data archive is also capable of creating a file that 
other [Daxa]{.smallcaps} users can import, and which will recreate the data archive by downloading the same data, and 
processing it in the same way; this renders making fully processed, and large, X-ray data files available with a piece 
of research unnecessary. This feature in particular can be used to further one of the tenets of open-source 
science - reproducibility. 


# Existing software packages
There are no direct analogues to our module, though we must acknowledge the many pieces of software (and data
archives), that greatly facilitated the development of [Daxa]{.smallcaps}. Data access is made 
possible primarily by the HEASARC data archive, though the Astroquery [@astroquery] package is also used. 
HEASARC provides an online interface to query their data archive, which has similar functionality to some of the 
filtering methods of mission classes in [Daxa]{.smallcaps} (though we provide slightly more functionality in 
that regard), and they provide Python SQL examples to access the data, but none of the data management and cleaning 
functionality that we include. 

[Daxa]{.smallcaps} also builds on the various telescope-specific software packages to perform data preparation and 
cleaning. Particularly important are the _XMM_ Science Analysis System (SAS; @sas) and the complementary extended 
SAS (eSAS; @esascook) packages, which allow us to provide simple Python interfaces to the complex, multi-step, processes 
that are required to prepare raw _XMM_ data for scientific use. The analogous _eROSITA_ Science Analysis Software 
System (eSASS; @erosita) must also be mentioned, as it provides the tools needed to reduce and prepare _eROSITA_ 
data. In this vein we must also acknowledge the HEASoft package, which is almost ubiquitous in X-ray data 
analyses, and is used by both SAS and eSASS. 

Another related software package is the other module in our open-source X-ray astronomy ecosystem, X-ray: Generate 
and Analyse ([Xga]{.smallcaps}; @xga) - it exists to analyse large samples of sources using large sets of X-ray 
data. [Daxa]{.smallcaps} is designed to go hand-in-hand with [Xga]{.smallcaps}, as it will build and manage the kind of 
dataset required for [Xga]{.smallcaps} to attain maximum usefulness. We emphasise that such datasets do not _have_ to 
be analysed with [Xga]{.smallcaps}) however. 

We have created a one-stop-shop for downloading and processing archival X-ray data, making it more accessible and 
user-friendly, particularly for non-specialists. [Daxa]{.smallcaps} is greater than the sum of its parts, but is only 
possible because of the existing software packages it builds upon; we hope that it only enhances the value that 
astrophysicists derive from the other software we have mentioned.

# Research projects using DAXA

[Daxa]{.smallcaps} has been useful for several research projects by different collaborations, and we anticipate that 
the number of research projects benefiting from its features will increase significantly. The primary use to which 
[Daxa]{.smallcaps} has been put is to assemble the multi-mission X-ray dataset (_XMM_, _Chandra_, _eROSITA_, _Swift_, 
and _ROSAT_) for the X-ray follow-up component of the Local Volume Complete Cluster Survey (LoVoCCS; @lovoccsI, @xlovoccs). It was used to 
identify the relevant observations, download, and process them, as well as to organise the significant number of files 
and make it easier for the dataset to be served to the X-ray community. Construction and administration of such 
large, complicated, multi-mission datasets is rendered quick and easy.

The X-ray Cluster Survey (XCS; formerly known as the _XMM_ Cluster Survey) collaboration now uses [Daxa]{.smallcaps} to
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
quasars [@ovigroups]. Absorption features that indicate the presence of Oxygen VI were identified in the spectra of 
several of the quasars, and the data that [Daxa]{.smallcaps} identified and retrieved allowed for an exploration of 
the hot-gas properties of these groups.

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

DT is grateful to Amanda Witwer for comments on the structure and content of this paper. 

# References
