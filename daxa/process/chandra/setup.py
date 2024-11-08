#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 08/11/2024, 12:48. Copyright (c) The Contributors

import os

from astropy.io import fits
from astropy.table import Table

from daxa.archive import Archive
from daxa.exceptions import NoProcessingError
from daxa.process.chandra._common import _ciao_process_setup


def parse_oif(oif_path: str):
    """
    A function that takes a path to a Chandra 'oif.fits' file included in each ObsID directory on the remote
    archive. The file will be filtered and parsed so that data relevant to DAXA processing valid scientific
    observations can be extracted. This includes things like which mode the detector was in, whether a grating
    was deployed, etc.

    The information will be simpler than what we tend to retrieve for XMM observations, as only one instrument is
    deployed in a particular observation.

    :param str oif_path: The path to the Chandra 'oif.fits' file that is to be parsed into a dictionary
        of relevant information and returned.
    :return: Multi-level dictionary of information.
    :rtype: dict
    """

    # Firstly, lets get this observation index file loaded in, so we can start reading out the important information
    oif_file = fits.open(oif_path)

    # We want both the header and the table - the header will give us some instrument info
    oif_hdr = oif_file[1].header
    # Convert to pandas because I prefer working with dataframes
    oif_tbl = Table(oif_file[1].data).to_pandas()
    # We're gonna make good use of the MEMBER_CONTENT column, but we wish to strip out the whitespace that can
    #  be present there
    oif_tbl['MEMBER_CONTENT'] = oif_tbl['MEMBER_CONTENT'].str.strip()

    # This feels somehow wrong, but we're just going to pull out the header keywords that I know are relevant D: Won't
    #  do a huge load of individual commands though, we'll set up a one-liner. We'll also set it up so the original
    #  header names can be converted to another name if we want to (None means the original name will be kept).
    hdr_to_store = {'SEQ_NUM': 'SEQUENCE', 'INSTRUME': None, 'DETNAM': 'DETECTOR', 'GRATING': None,
                    'OBS_MODE': None, 'DATAMODE': 'MODE', 'RA_NOM': None, 'DEC_NOM': None, 'ROLL_NOM': None,
                    'SIM_X': 'SCIENCE_INST_MODULE_X', 'SIM_Y': 'SCIENCE_INST_MODULE_Y',
                    'SIM_Z': 'SCIENCE_INST_MODULE_Z'}
    rel_hdr_info = {hdr_key if new_key is None else new_key: oif_hdr[hdr_key]
                    for hdr_key, new_key in hdr_to_store.items()}

    # ------------------- ANY MODIFICATION OF HEADER DATA HAPPENS HERE ------------------
    rel_hdr_info['DETECTOR'] = rel_hdr_info['DETECTOR'].split('-')[-1]
    # Separate the GRATING info into a simple boolean look-up as to whether it has one deployed or not, and another
    #  header containing the name
    rel_hdr_info['GRATING_NAME'] = '' if rel_hdr_info['GRATING'] == 'NONE' else rel_hdr_info['GRATING']
    rel_hdr_info['GRATING'] = False if rel_hdr_info['GRATING'] == 'NONE' else True
    # -----------------------------------------------------------------------------------

    # This temporarily stores relevant quantities we derive from the file table
    rel_tbl_info = {}
    # Now we move to examining the data file table - first off we set an active value by checking if a processed
    #  event list exists
    rel_tbl_info['active'] = 'EVT2' in oif_tbl['MEMBER_CONTENT'].values
    # Quickly count the number of times each type of file is present, and convert to a dictionary
    mem_type_cnts = oif_tbl['MEMBER_CONTENT'].str.strip().value_counts().to_dict()

    alt_exp_mode = False
    sub_exp = False
    # Then we use them to try and determine if the data were taken in some unusual observing modes - we use the
    #  EVT1 count as a trigger because multi-OBI observations can combine their multiple exposures into a single
    #  EVT2 event list
    if 'EVT1' in mem_type_cnts and mem_type_cnts['EVT1'] > 1:
        # First we look for 'alternating exposure mode', which would result in multiple event lists, one with e1 in
        #  the name and another with e2 in the name
        if (oif_tbl['MEMBER_LOCATION'].str.contains('_e2_').any() and
                oif_tbl['MEMBER_LOCATION'].str.contains('_e1_').any()):
            alt_exp_mode = True

        # Now we arrive at 'multiple observation intervals', which seem equivalent to sub-exposures in the
        #  XMM world (you can tell what X-ray telescope I 'grew up with' academically speaking). In the Chandra
        #  archive they seem incredibly rare, at the time of writing the docs page
        #  (https://cxc.harvard.edu/ciao/why/multiobi.html) only mentioned twenty ObsIDs
        else:
            sub_exp = True
    elif 'EVT1' not in mem_type_cnts:
        # This happens for ObsID 94 and ObsID 107 - there are no EVT1 files listed in their OIF, and so I think
        #  we just have to set them to inactive
        rel_tbl_info['active'] = False

    # Add them into the information dictionary
    rel_tbl_info['alt_exp_mode'] = alt_exp_mode
    rel_tbl_info['sub_exp'] = sub_exp

    # We're also going to extract the sub-exposure IDs (multi-OBI mode is very rarely used it seems, but if I
    #  want to support those observations, which I do, then we need to read them out). We also need to set
    #  a sub-exposure ID in the case where it Chandra isn't in multi-OBI mode, as the infrastructure demands
    # that if one ObsID has sub-exposures then they all do
    if sub_exp:
        evt1_files = oif_tbl[oif_tbl['MEMBER_CONTENT'] == 'EVT1']['MEMBER_LOCATION'].values
        sub_exp_ids = ['E' + ev1_f.split('_')[-2].split('N')[0] for ev1_f in evt1_files]
        sub_exp_ids.sort()
    else:
        sub_exp_ids = ['E001']
    # Then that list of sub-exposure IDs (or single ID in most cases) is stored in the output dictionary
    rel_tbl_info['sub_exp_ids'] = sub_exp_ids

    # Now check to make sure that there are aspect solution files included somewhere - there are very small number
    #  of observations which don't seem to have them, and probably can't be reprocessed (see issue #345), so
    #  they are set as inactive
    if 'ASPSOL' not in mem_type_cnts and 'ASPSOLOBI' not in mem_type_cnts:
        rel_tbl_info['active'] = False

    # We're also going to store the counts of how many of each type of file are present - it might be useful later
    rel_tbl_info['file_content_counts'] = mem_type_cnts

    # ------------------- HERE WE CONSTRUCT THE RETURN DICTIONARY -------------------
    # The observation_summaries property of Archive expects the level below ObsID to be instrument names, or
    #  in this case just the one instrument name
    obs_info = {rel_hdr_info['INSTRUME']: None}
    # Drop the INSTRUME entry from rel_hdr_info now, we don't need it there any longer
    inst = rel_hdr_info.pop('INSTRUME')
    obs_info[inst] = rel_hdr_info
    obs_info[inst].update(rel_tbl_info)
    # -------------------------------------------------------------------------------

    return obs_info


def prepare_chandra_info(archive: Archive):
    """
    A simple function that runs through all Chandra observations, reads through their file inventory, and includes
    that information in the Archive in a parsed, standardized form.

    :param Archive archive: A DAXA archive containing a Chandra mission.
    """

    # Check that the archive in question has got Chandra data etc.
    ciao_vers, caldb_vers, chandra_miss = _ciao_process_setup(archive, make_dirs=True)

    # This very simply iterates through the Chandra missions, and through all their ObsIDs, and parses the
    #  observation index files (assuming that is what 'oif' stands for?)
    obs_sums = {}
    proc_succ = {}
    proc_errs = {}
    # Beyond this point these dictionaries will be blank (at the time of designing at least), but are necessary
    #  for the observation archive to feel good about itself
    proc_logs = {}
    proc_einfo = {}
    proc_conf = {}
    for miss in chandra_miss:
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
            check_dict = archive.process_success[miss.name]['prepare_chandra_info']
        except (NoProcessingError, KeyError):
            check_dict = {}

        for oi in miss.filtered_obs_ids:
            # If there has been a run of this function for this ObsID before, we can skip it
            if oi in check_dict:
                continue
            try:
                # This sets up the absolute path to the 'oif.fits' file for the current Chandra mission and ObsID
                cur_path = os.path.join(miss.raw_data_path, oi, 'oif.fits')
                # The parsing function reads through that file and spits out the important information
                parsed_info = parse_oif(cur_path)
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
                proc_errs[miss.name][oi] = 'The OIF for Chandra observation {oi} cannot be found.'.format(oi=oi)
                proc_logs[miss.name][oi] = ''
                proc_einfo[miss.name][oi] = {}
                proc_conf[miss.name][oi] = ''

    # This just makes sure that at least one observation has an OIF file that we can use - if not then I think
    #  it is likely something has gone wrong on the backend, but whatever the reason we can't continue into any
    #  Chandra processing.
    for miss in chandra_miss:
        if all([len(en) == 0 for oi, en in obs_sums[miss.name].items()]):
            raise FileNotFoundError("No {} oif.fits files could be found to process.".format(miss.pretty_name))

    # Finally the fully populated dictionary is added to the archive - this will be what informs DAXA about
    #  which Chandra observations it can actually process into something usable
    archive.observation_summaries = obs_sums
    archive.process_success = ('prepare_chandra_info', proc_succ)
    archive.raw_process_errors = ('prepare_chandra_info', proc_errs)
    archive.process_logs = ('prepare_chandra_info', proc_logs)
    archive.process_extra_info = ('prepare_chandra_info', proc_einfo)
    archive.process_configurations = ('prepare_chandra_info', proc_conf)


def det_name_to_chip_ids():
    pass
    """
    0  I0, ACIS-I0	FI	w203c4r
    1  I1, ACIS-I1	FI	w193c2 
    2  I2, ACIS-I2	FI	w158c4r
    3  I3, ACIS-I3	FI	w215c2r
    4  S0, ACIS-S0	FI	w168c4r
    5  S1, ACIS-S1	BI	w140c4r
    6  S2, ACIS-S2	FI	w182c4r
    7  S3, ACIS-S3	BI	w134c4r
    8  S4, ACIS-S4	FI	w457c4 
    9  S5, ACIS-S5	FI	w201c3r
    
    CCD-ID-3 (I3) contains the ACIS-I aimpoint
    CCD-ID 7 (S3) contains the ACIS-S aimpoint
    """
    raise NotImplementedError("The conversion of detector name to a list of CCD IDs has not yet been implemented.")
