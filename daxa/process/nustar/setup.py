#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 08/11/2024, 15:37. Copyright (c) The Contributors

import os
from warnings import simplefilter

from astropy.io import fits
from astropy.table import Table

from daxa.archive import Archive
from daxa.exceptions import NoProcessingError
from daxa.process.nustar._common import _nustardas_process_setup

# This is because astropy gets quite upset by one of the headers in the NuSTAR observation catalog files
simplefilter('ignore', fits.card.AstropyUserWarning)


def parse_cat(cat_path: str):
    """
    A function that takes a path to a NuSTAR observation catalog file (included in each ObsID directory on the remote
    archive). The file will be filtered and parsed so that data relevant to DAXA processing valid scientific
    observations can be extracted - this should be quite simple for NuSTAR, as it is a relatively simple telescope

    :param str cat_path: The path to the NuSTAR catalog file that is to be parsed into a dictionary of relevant
        information and returned.
    :return: Observation summary dictionary - really simple in the case of NuSTAR.
    :rtype: dict
    """

    # Reading in the file as a fits HDUList first
    cat_file = fits.open(cat_path)

    # Read out the file table and convert to pandas because I prefer working with dataframes
    cat_tbl = Table(cat_file[1].data).to_pandas()

    # This is the dictionary that will be returned at the end - remember that there are two telescopes/instruments!
    obs_info = {'FPMA': {}, 'FPMB': {}}

    for inst in obs_info:
        # If there are no raw events then we have to assume the instrument was not active
        if any(cat_tbl['DESCRIP'].str.contains((inst + " raw unfiltered events"))):
            obs_info[inst]['active'] = True
        else:
            obs_info[inst]['active'] = False

    return obs_info


def prepare_nustar_info(archive: Archive):
    """
    A simple function that runs through all NuSTAR observations, reads through their file inventory, and includes
    that information in the Archive in a parsed, standardized form.

    This is a very similar implementation to the 'prepare_chandra_info' function.

    :param Archive archive: A DAXA archive containing a NuSTAR mission.
    """

    # Check that the archive in question has got NuSTAR data, software, etc.
    nudas_vers, caldb_vers, nustar_miss = _nustardas_process_setup(archive, make_dirs=True)

    # Iterates through NuSTAR missions, parses their observation file catalogs, and adds that information to the
    #  archive - that information will tell us if a particular observation is usable
    obs_sums = {}
    proc_succ = {}
    proc_errs = {}
    # Beyond this point these dictionaries will be blank (at the time of designing at least), but are necessary
    #  for the observation archive to feel good about itself
    proc_logs = {}
    proc_einfo = {}
    proc_conf = {}
    for miss in nustar_miss:
        # Add an entry for the current mission
        obs_sums.setdefault(miss.name, {})
        proc_succ.setdefault(miss.name, {})
        proc_errs.setdefault(miss.name, {})
        proc_logs.setdefault(miss.name, {})
        proc_einfo.setdefault(miss.name, {})
        proc_conf.setdefault(miss.name, {})

        # We want to see if this has been run before, and if it has are there any observations it has not been
        #  run for yet - as this is the first process in the chain, we need to account for the fact that nothing
        #  has been run before, and using the process_success property might raise an exception
        try:
            check_dict = archive.process_success[miss.name]['prepare_nustar_info']
        except (NoProcessingError, KeyError):
            check_dict = {}

        for oi in miss.filtered_obs_ids:
            # If there has been a run of this function for this ObsID before, we can skip it
            if oi in check_dict:
                continue
            try:
                # This sets up the absolute path to the 'oif.fits' file for the current NuSTAR mission and ObsID
                cur_path = os.path.join(miss.raw_data_path, oi, '{}_cat.fits'.format(oi))
                # The parsing function reads through that file and spits out the important information
                parsed_info = parse_cat(cur_path)
                # Then we add it to the dictionary
                obs_sums[miss.name][oi] = parsed_info
                # We can then say that this process for this ObsID was a success
                proc_succ[miss.name][oi] = True
                # Maybe this will add something one day, but doesn't right now
                proc_errs[miss.name][oi] = ''
                proc_logs[miss.name][oi] = ''
                proc_einfo[miss.name][oi] = {}
                proc_conf[miss.name][oi] = ''

            except FileNotFoundError:
                # Here though, something unfortunate has gone wrong - we'll warn them that the file can't be
                #  found and store that this process failed
                proc_succ[miss.name][oi] = False
                obs_sums[miss.name][oi] = {}
                # We do store an error as well, just so they know
                proc_errs[miss.name][oi] = ('The observation file catalog for NuSTAR observation {oi} cannot be '
                                            'found.').format(oi=oi)
                proc_logs[miss.name][oi] = ''
                proc_einfo[miss.name][oi] = {}
                proc_conf[miss.name][oi] = ''

    # This just makes sure that at least one observation has an OIF file that we can use - if not then I think
    #  it is likely something has gone wrong on the backend, but whatever the reason we can't continue into any
    #  Chandra processing.
    for miss in nustar_miss:
        if all([len(en) == 0 for oi, en in obs_sums[miss.name].items()]):
            raise FileNotFoundError("No {} observation catalog files could be found to "
                                    "process.".format(miss.pretty_name))

    # Finally the fully populated dictionary is added to the archive - this will be what informs DAXA about
    #  which NuSTAR observations it can actually process into something usable
    archive.observation_summaries = obs_sums
    archive.process_success = ('prepare_nustar_info', proc_succ)
    archive.raw_process_errors = ('prepare_nustar_info', proc_errs)
    archive.process_logs = ('prepare_nustar_info', proc_logs)
    archive.process_extra_info = ('prepare_nustar_info', proc_einfo)
    archive.process_configurations = ('prepare_nustar_info', proc_conf)
