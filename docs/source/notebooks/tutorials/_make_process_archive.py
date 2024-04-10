#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 10/04/2024, 15:36. Copyright (c) The Contributors

# This script will create and process (XMM process anyway) an archive of observations of the quasar PHL 1811, so I
#  have something to load in for the archives tutorial

from daxa.archive import Archive
from daxa.mission import XMMPointed, Chandra, NuSTARPointed, ROSATAllSky
from daxa.process.simple import full_process_xmm

xm = XMMPointed()
ch = Chandra()
ra = ROSATAllSky()
nu = NuSTARPointed()

xm.filter_on_name("PHL 1811")
bodge = xm.filter_array.copy()
# This is a bodge - I'm doing this for illustrative purposes, neither of these observations has anything to do with the
#  PHL 1811 - 0502671101 will have one sub-exposure fail the espfilt processing step (I think it was) and 0102041001
#  is entirely CalClosed observations so everything will fail
bodge[xm.all_obs_info['ObsID'].isin(['0502671101', '0102041001'])] = True
xm.filter_array = bodge
ch.filter_on_name("PHL 1811")
ra.filter_on_name("PHL 1811")
nu.filter_on_name("PHL 1811")

xm.download()
ch.download(download_products=True)
ra.download(download_products=True)
nu.download(download_products=True)

arch = Archive("PHL1811_made_earlier", [xm, ch, nu, ra])

full_process_xmm(arch)
