#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 11/04/2024, 11:39. Copyright (c) The Contributors

# This script will create and process (XMM process anyway) an archive of observations of the quasar PHL 1811, so I
#  have something to load in for the archives tutorial

import os

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
#  is entirely CalClosed observations so everything will fail. The 0105261001 ObsID has unscheduled sub-exposures as
#  well
bodge[xm.all_obs_info['ObsID'].isin(['0502671101', '0102041001', '0105261001'])] = True
xm.filter_array = bodge
ch.filter_on_name("PHL 1811")
ra.filter_on_name("PHL 1811")
nu.filter_on_name("PHL 1811")

xm.download()
ch.download(download_products=True)
ra.download(download_products=True)
nu.download(download_products=True)

arch = Archive("PHL1811_made_earlier", [xm, ch, nu, ra], clobber=True)

full_process_xmm(arch)

# Listing the region files in the test directory
reg_files = os.listdir('region_files')

# Setting up the structure of the dictionary we will pass to the archive at the end of this
reg_paths = {'xmm_pointed': {}}
# Iterating through the ObsIDs in the XMMPointed mission
for oi in arch['xmm_pointed'].filtered_obs_ids:
    # Checking to see which have a corresponding region file
    if any([oi in rf for rf in reg_files]):
        # Generating the path to the image we need for pixel to RA-Dec conversion
        im_pth = arch.get_current_data_path('xmm_pointed', oi) + \
        'images/{}_mos1_0.5-2.0keVimg.fits'.format(oi)
        # Setting up the entry in the final dictionary, with the path to the regions and the image
        reg_paths['xmm_pointed'][oi] = {'region': 'region_files/{}.reg'.format(oi), 'wcs_src': im_pth}

# Adding the regions to the archive
arch.source_regions = reg_paths
