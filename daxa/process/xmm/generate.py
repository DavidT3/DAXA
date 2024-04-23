#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 23/04/2024, 15:45. Copyright (c) The Contributors
import os
import shutil
from typing import Tuple
from warnings import warn

import numpy as np
import pandas as pd
import xga
from astropy.units import Quantity, UnitConversionError
from xga.sas import eexpmap
from xga.sources import NullSource

from daxa import NUM_CORES
from daxa.archive.base import Archive
from daxa.process.xmm._common import ALLOWED_XMM_MISSIONS


def _en_checks(lo_en: Quantity, hi_en: Quantity) -> Tuple[Quantity, Quantity]:
    """
    This internal functions performs the checks on low and high energy bounds that are common to multiple generation
    functions such as those that generate imaged and those that generate exposure maps.

    :param Quantity lo_en: The lower energy bound(s) for the product being generated. This can either be passed as a
        scalar Astropy Quantity or, if sets of the same product in different energy bands are to be generated, as a
        non-scalar Astropy Quantity. If multiple lower bounds are passed, they must each have an entry in the
        hi_en argument.
    :param Quantity hi_en: The upper energy bound(s) for the product being generated. This can either be passed as a
        scalar Astropy Quantity or, if sets of the same product in different energy bands are to be generated, as a
        non-scalar Astropy Quantity. If multiple upper bounds are passed, they must each have an entry in the
        lo_en argument.
    :return: The lower and upper energy bound arguments, converted to keV.
    :rtype: Tuple[Quantity, Quantity]
    """

    if not lo_en.unit.is_equivalent('keV') or not hi_en.unit.is_equivalent('keV'):
        raise UnitConversionError("Both the lo_en and hi_en arguments must be in units that can be converted to keV.")
    # Multiple lower energy and upper energy bounds can be passed, if multiple sets of images are to be generated. Here
    #  we check that if multiple lower energy bounds have been passed, multiple upper bounds have been
    #  too (and vica versa)
    elif lo_en.isscalar != hi_en.isscalar:
        raise ValueError("If either lo_en or hi_en is scalar (one value), the other energy bound argument must be too.")
    # At this point we're sure that both energy bounds are either scalar or not, so just checking for non-scalarness
    #  with one of the bounds is fine - we now need to make sure that the same number of bounds are in each parameter
    elif not lo_en.isscalar and len(lo_en) != len(hi_en):
        raise ValueError("The lo_en argument has {le} entries, and the hi_en argument has {he} entries - the number "
                         "of entries must be the same.".format(le=len(lo_en), he=len(hi_en)))
    else:
        # Make sure that the energies are in keV
        lo_en = lo_en.to('keV')
        hi_en = hi_en.to('keV')

    # Next we make sure that if only one pair of energy bounds have been passed, we can still iterate through them
    if lo_en.isscalar:
        lo_en = np.array([lo_en])
        hi_en = np.array([hi_en])

    # Finally, need to check that the lower energy bounds are all smaller than the upper, anything else
    #  would be unphysical.
    if not (lo_en < hi_en).all():
        raise ValueError("All lower energy bounds (lo_en) must be smaller than their upper bound (hi_en) equivalent.")

    return lo_en, hi_en


def generate_images_expmaps(obs_archive: Archive, lo_en: Quantity = Quantity([0.5, 2.0], 'keV'),
                            hi_en: Quantity = Quantity([2.0, 10.0], 'keV'), num_cores: int = NUM_CORES):
    """
    A function to generate images and exposure maps for a processed XMM mission dataset contained within an
    archive. Users can select the energy band(s) that they wish to generate images and exposure maps within.

    :param Archive obs_archive:
    :param lo_en: The lower energy bound(s) for the product being generated. This can either be passed as a
        scalar Astropy Quantity or, if sets of the same product in different energy bands are to be generated, as a
        non-scalar Astropy Quantity. If multiple lower bounds are passed, they must each have an entry in the
        hi_en argument. The default is 'Quantity([0.5, 2.0], 'keV')', which will generate two sets of products, one
        with lower bound 0.5 keV, the other with lower bound 2 keV.
    :param hi_en: The upper energy bound(s) for the product being generated. This can either be passed as a
        scalar Astropy Quantity or, if sets of the same product in different energy bands are to be generated, as a
        non-scalar Astropy Quantity. If multiple upper bounds are passed, they must each have an entry in the
        lo_en argument. The default is 'Quantity([2.0, 10.0], 'keV')', which will generate two sets of products, one
        with upper bound 2.0 keV, the other with upper bound 10 keV.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    """

    # Run the energy bounds checks
    _en_checks(lo_en, hi_en)

    # Just grabs the XMM missions, we already know there will be at least one because otherwise _sas_process_setup
    #  would have thrown an error
    xmm_miss = [mission for mission in obs_archive if mission.name in ALLOWED_XMM_MISSIONS]
    # We are iterating through XMM missions (options could include xmm_pointed and xmm_slew for instance).
    for miss in xmm_miss:
        # This will trigger and exception if the mission is for slew data, which XGA doesn't work with at
        #  the time of writing.
        if miss == 'xmm_slew':
            raise NotImplementedError("XGA cannot currently be used to generate XMM slew observation images.")

        # Identifiers for all the valid data are fetched, and will be narrowed down later so that only those which
        #  had cleaned event lists generated successfully are selected. For this check I know I only want the
        #  ObsIDs and instruments - and also that there shouldn't be duplicates
        rel_obs_info = np.array(obs_archive.get_obs_to_process(miss.name))[:, :2]
        # This ensures the rows of rel_obs_info are unique
        rel_obs_info = np.unique(rel_obs_info, axis=0)

        # If the merging function hasn't been run, I won't allow this function to run - I was thinking I would let
        #  the user generate sub-exposure images, but actually XGA doesn't support that, and though it would be
        #  possible by iterating and bodging, I'm not going to do that now
        good_ce = obs_archive.check_dependence_success(miss.name, rel_obs_info, 'merge_subexposures')
        val_obs_info = np.array(rel_obs_info)[good_ce]

        # If there are no entries in rel_ids then there are no event lists to work on, so we raise a warning
        if len(val_obs_info) == 0:
            warn("Every merge_subexposures run for the {m} mission in the {a} archive is reporting as a "
                 "failure, skipping process.".format(m=miss.name, a=obs_archive.archive_name), stacklevel=2)
            continue

        # The dictionary that stores which ObsIDs have which instruments
        which_obs = {}
        # A dictionary that stores example file names for event lists - these will be overwritten every iteration
        #  but that's okay.
        evt_names = {}
        # A bit ugly and inelegant but oh well - this just iterates through all the relevant IDs, separating them
        #  into a dictionary that has ObsIDs as top level keys, and a list of available instruments as the values
        for obs_info in val_obs_info:
            # Unpacking the obs_info
            obs_id, inst = obs_info
            # Combining the info for a current ID to access merged event lists with
            cur_id = ''.join(obs_info)

            if obs_id not in which_obs:
                which_obs[obs_id] = [inst]
            else:
                which_obs[obs_id].append(inst)

            # Grabs the output path for the final event list, then splits it to remove the absolute bit of the
            #  absolute path, leaving just the filename.
            evt_path = obs_archive.process_extra_info[miss.name]['merge_subexposures'][cur_id]['final_evt']
            evt_names[inst] = evt_path.split('/')[-1].replace(obs_id, '{obs_id}')

        # It is conceivable that there are no observations with a particular instrument, so we check for that and
        #  put a dummy path in the dictionary because whilst it may need to actually point at a file, XGA does need
        #  an entry in the configuration file for all instruments
        if 'PN' not in evt_names:
            evt_names['PN'] = 'PN_clean_evts.fits'
        if 'M1' not in evt_names:
            evt_names['M1'] = 'M1_clean_evts.fits'
        if 'M2' not in evt_names:
            evt_names['M2'] = 'M2_clean_evts.fits'

        # Now we can do some post-processing, and turn the information in 'which_obs' into the various
        #  dataframes required to trick XGA into making our images for us. Normally you have to setup a configuration
        #  file to use XGA, that points it to where the event lists etc. live, but here we're going to alter the
        #  XGA variables that tell the module where to look for data AFTER it's been imported

        # These are the column names for an XGA census - will be used in the creation of a pandas dataframe
        census_cols = ['ObsID', 'RA_PNT', 'DEC_PNT', 'USE_PN', 'USE_MOS1', 'USE_MOS2']
        # This list will get filled in with the observation data
        census_data = []
        for obs_id in which_obs:
            # Grabs the information about the current ObsID, which we will use to supply the pointing coordinates
            obs_info = obs_archive[miss.name].all_obs_info[obs_archive[miss.name].all_obs_info['ObsID'] ==
                                                           obs_id].iloc[0]
            census_data.append([obs_id, obs_info['ra'], obs_info['dec'], 'PN' in which_obs[obs_id],
                                'M1' in which_obs[obs_id], 'M2' in which_obs[obs_id]])
        # Constructs the census dataframe from the data we assembled, and the columns we defined earlier
        census = pd.DataFrame(census_data, columns=census_cols)

        # Makes an empty blacklist dataframe - we don't want to blacklist any observation, so we have to replace
        #  whatever XGA already had loaded
        blacklist_cols = ["ObsID", "EXCLUDE_PN", "EXCLUDE_MOS1", "EXCLUDE_MOS2"]
        blacklist = pd.DataFrame(None, columns=blacklist_cols)

        # Setting up the configuration file - all that really matters in this is the paths to the cleaned event
        #  lists, and the path to the attitude file.
        # TODO set the attitude file programmatically
        xmm_files = {"root_xmm_dir": obs_archive.archive_path+'processed_data/' + miss.name + '/',
                     "clean_pn_evts": obs_archive.archive_path+'processed_data/' + miss.name + '/{obs_id}/events/' +
                                      evt_names['PN'],
                     "clean_mos1_evts": obs_archive.archive_path+'processed_data/' + miss.name + '/{obs_id}/events/' +
                                        evt_names['M1'],
                     "clean_mos2_evts": obs_archive.archive_path+'processed_data/' + miss.name + '/{obs_id}/events/' +
                                        evt_names['M2'],
                     "attitude_file": obs_archive.archive_path+'processed_data/' + miss.name +
                                      '/{obs_id}/P{obs_id}OBX000ATTTSR0000.FIT',
                     "lo_en": ['0.50', '2.00'],
                     "hi_en": ['2.00', '10.00'],
                     "pn_image": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-pn_merged_img.fits",
                     "mos1_image": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-mos1_merged_img.fits",
                     "mos2_image": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-mos2_merged_img.fits",
                     "pn_expmap": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-pn_merged_img.fits",
                     "mos1_expmap": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-mos1_merged_expmap.fits",
                     "mos2_expmap": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-mos2_merged_expmap.fits",
                     "region_file": "/this/is/optional/xmm_obs/regions/{obs_id}/regions.reg"}

        # The actual config file has another subdictionary, but that doesn't matter for this bodge
        xmm_config = {'XMM_FILES': xmm_files}

        # This is where the outputs from XGA will be stored
        new_out = obs_archive.archive_path+'processed_data/' + miss.name + '/xga_output/'
        # This makes sure that the directory exists, if it doesn't already
        if not os.path.exists(new_out):
            os.makedirs(new_out)

        # Here we manually replace the global variables in XGA - this is viable because the NullSource class is very
        #  limited and we're using it to do a very limited number of things. Otherwise replacing variables like this
        #  in a module is a very bad idea - you'll see that some variables have to be overwritten multiple times in
        #  different places, and that's because the sub-modules load in the CENSUS, OUTPUT etc. global variables from
        #  utils when they're imported, and if this function is called multiple times for multiple archives in one
        #  session then we have to overwrite the OUTPUT variable in those individual sub-modules.
        xga.sources.base.CENSUS = census
        xga.sources.base.xga_conf = xmm_config
        xga.sources.base.BLACKLIST = blacklist
        xga.sources.base.OUTPUT = new_out
        xga.sas.phot.OUTPUT = new_out
        xga.sas.misc.OUTPUT = new_out

        # Setting up a NullSource, which will contain every ObsID in this archive
        null_src = NullSource()

        # Iterating through the energy band pairs (if there are multiple), to generate images and exposure maps.
        for en_ind, lo in enumerate(lo_en):
            hi = hi_en[en_ind]
            # This function will also generate images, before it makes exposure maps
            eexpmap(null_src, lo, hi, num_cores=num_cores)

        # This goes through and
        for obs_id in which_obs:
            # This is the directory where XGA stored the files generated for the current value of ObsID
            cur_path = new_out + obs_id + '/'

            # We set up the path we're going to move things too, in the existing DAXA directory structure - file
            #  names will be added onto the end
            dest_dir = obs_archive.construct_processed_data_path(miss.name, obs_id) + 'images/'

            # We make sure that directory exists (can't think why it wouldn't, but better to be safe).
            if os.path.exists(cur_path):
                for file_name in os.listdir(cur_path):

                    # We convert them to the new DAXA naming convention for files
                    if 'img' in file_name:
                        inst = file_name.split('_')[1]
                        inst = miss.check_inst_names(inst, error_on_bad_inst=False, show_warn=False)[0]

                        cur_lo, cur_hi = file_name.split("_")[-1].split('keV')[0].split("-")
                        new_name = "obsid{oi}-inst{i}-subexpALL-en{l}_{h}keV-image.fits".format(oi=obs_id, i=inst,
                                                                                                l=cur_lo, h=cur_hi)
                    elif 'expmap' in file_name:
                        inst = file_name.split('_')[1]
                        inst = miss.check_inst_names(inst, error_on_bad_inst=False, show_warn=False)[0]

                        cur_lo, cur_hi = file_name.split("_")[-1].split('keV')[0].split("-")
                        new_name = "obsid{oi}-inst{i}-subexpALL-en{l}_{h}keV-expmap.fits".format(oi=obs_id, i=inst,
                                                                                                 l=cur_lo, h=cur_hi)
                    # Move the file to its new home, with its new name
                    dest_file_path = dest_dir + new_name

                    # Doing the actual moving of the directory
                    shutil.move(cur_path + file_name, dest_file_path)

        # Finally we remove the XGA output directory.
        shutil.rmtree(new_out)




