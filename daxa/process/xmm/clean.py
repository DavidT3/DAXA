#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 16/12/2022, 13:42. Copyright (c) The Contributors
from typing import Union
from warnings import warn

from astropy.units import Quantity, UnitConversionError

from daxa import NUM_CORES
from daxa.archive.base import Archive
from daxa.exceptions import NoDependencyProcessError
from daxa.process.xmm._common import _sas_process_setup, ALLOWED_XMM_MISSIONS


def espfilt(obs_archive: Archive, method: str = 'histogram', with_smoothing: Union[None, bool, Quantity] = False,
            with_binning: Union[None, bool, Quantity] = False, ratio: float = 1.2,
            lo_en: Quantity = Quantity(2500, 'eV'), hi_en: Quantity = Quantity(8000, 'eV'), range_scale: dict = None,
            allowed_sigma: float = 2.5, gauss_fit_lims: tuple = (0.1, 6.5), num_cores: int = NUM_CORES,
            disable_progress: bool = False):
    # Run the setup for SAS processes, which checks that SAS is installed, checks that the archive has at least
    #  one XMM mission in it, and shows a warning if the XMM missions have already been processed
    sas_version = _sas_process_setup(obs_archive)

    # Checking to make sure an acceptable value for 'method' has been passed
    if method != 'histogram' and method != 'ratio':
        raise ValueError("The string passed for 'method' must be either histogram or ratio.")

    # Parsing the user's choice of parameter for espfilt smoothing
    if isinstance(with_smoothing, bool) and with_smoothing:
        smooth_factor = Quantity(51, 's')
    elif isinstance(with_smoothing, bool) and not with_smoothing:
        smooth_factor = None
    elif with_smoothing is None:
        smooth_factor = None
    elif isinstance(with_smoothing, Quantity) and not with_smoothing.unit.is_equivalent('s'):
        raise UnitConversionError("If the 'with_smoothing' argument is an Astropy Quantity, it must be in units that"
                                  " can be converted to seconds.")
    elif with_smoothing < Quantity(1, 's') or with_smoothing > Quantity(60, 's'):
        raise ValueError("If the with_smoothing argument is set (to activate light-curve smoothing) then it must be "
                         "set to a value greater than 1 second and less than 60 seconds.")
    elif isinstance(with_smoothing, Quantity):
        smooth_factor = with_smoothing.to('s')

    # Parsing the user's choice of parameters for espfilt binning
    if isinstance(with_binning, bool) and with_binning:
        bin_size = Quantity(60, 's')
    elif isinstance(with_binning, bool) and not with_binning:
        bin_size = None
    elif with_binning is None:
        bin_size = None
    elif isinstance(with_binning, Quantity) and not with_binning.unit.is_equivalent('s'):
        raise UnitConversionError("If the 'with_binning' argument is an Astropy Quantity, it must be in units that"
                                  " can be converted to seconds.")
    elif isinstance(with_binning, Quantity) and with_binning < Quantity(1, 's'):
        raise ValueError("If the with_binning argument is set (to activate light-curve binning) then it must be set "
                         "to a value greater than 1 second.")
    elif isinstance(with_binning, Quantity):
        bin_size = with_binning.to('s')

    if not isinstance(ratio, (int, float)):
        raise TypeError("The ratio argument must be either a float or an integer.")

    if not lo_en.unit.is_equivalent('eV') or not hi_en.unit.is_equivalent('eV'):
        raise UnitConversionError("The lo_en and hi_en arguments must be astropy quantities in units that can be "
                                  "converted to eV.")
    elif lo_en <= hi_en:
        raise ValueError("The hi_en argument must be larger than the lo_en argument.")
    else:
        lo_en = lo_en.to('eV')
        hi_en = hi_en.to('eV')

    if (lo_en < Quantity(1, 'eV') or lo_en > Quantity(32766, 'eV')) or \
            (hi_en < Quantity(2, 'eV') or hi_en > Quantity(32767, 'eV')):
        raise ValueError("The lo_en value must be greater than 1 eV and less than 32766 eV, the hi_en value must be "
                         "greater than 2 eV and lower than 32767 eV.")

    # Setting the default values for range scale, can't have a mutable argument
    if range_scale is None:
        range_scale = {'mos': 6.0, 'pn': 15.0}
    elif range_scale is not None and not isinstance(range_scale, dict):
        raise ValueError("The range_scale argument must be a dictionary with an entry for MOS instruments (with key "
                         "'mos') and an entry for the PN instrument (with key 'pn').")
    elif range_scale is not None and isinstance(range_scale, dict) and (
            'pn' not in range_scale or 'mos' not in range_scale):
        raise KeyError("The range_scale argument must be a dictionary with an entry for MOS instruments (with key "
                       "'mos') and an entry for the PN instrument (with key 'pn').")

    if not isinstance(allowed_sigma, (int, float)):
        raise TypeError("The allowed_sigma argument must be either an integer or a float.")

    if not isinstance(gauss_fit_lims, tuple):
        raise TypeError("The gauss_fit_lims argument must be a tuple.")
    elif len(gauss_fit_lims) != 2:
        raise ValueError("The gauss_fit_lims tuple must have two elements; the first the lower limit, and the second "
                         "the upper limit.")
    elif not all([isinstance(g_lim, (float, int)) for g_lim in gauss_fit_lims]):
        raise TypeError("The elements of gauss_fit_lims must be either integers or floats.")

    # Define the form of the emchain command that must be run to check for anomalous states in MOS CCDs
    ef_cmd = "cd {d}; export SAS_CCF={ccf}; espfilt eventfile={ef} withoot={woot} ootfile={oot} method={me} " \
             "withsmoothing={ws} smooth={s} withbinning={wb} binsize={bs} ratio={r} withlongnames=yes elow={el} " \
             "ehigh={eh} rangescale={rs} allowsigma={as} limits={ls}"

    # Sets up storage dictionaries for bash commands, final file paths (to check they exist at the end), and any
    #  extra information that might be useful to provide to the next step in the generation process
    miss_cmds = {}
    miss_final_paths = {}
    miss_extras = {}

    # Just grabs the XMM missions, we already know there will be at least one because otherwise _sas_process_setup
    #  would have thrown an error
    xmm_miss = [mission for mission in obs_archive if mission.name in ALLOWED_XMM_MISSIONS]
    # We are iterating through XMM missions (options could include xmm_pointed and xmm_slew for instance).
    for miss in xmm_miss:
        # Sets up the top level keys (mission name) in our storage dictionaries
        miss_cmds[miss.name] = {}
        miss_final_paths[miss.name] = {}
        miss_extras[miss.name] = {}

        # Need to check to see whether ANY of ObsID-instrument-subexposure combos have had emchain run for them, as
        #  it is a requirement for this processing function. There will probably be a more elegant way of checkinf
        #  at some point in the future, generalised across all SAS functions
        if 'emchain' not in obs_archive.process_success[miss.name] \
                and 'epchain' not in obs_archive.process_success[miss.name]:
            raise NoDependencyProcessError("Neither the emchain (for MOS) nor the epchain (for PN) step has been run "
                                           "for the {m} mission in the {a} archive. At least one of these must have "
                                           "been run to use espfilt.".format(m=miss.name, a=obs_archive.archive_name))
        # If every emchain run was a failure then we warn the user and move onto the next XMM mission (if there
        #  is one).
        elif ('emchain' in obs_archive.process_success[miss.name] and all(
                [v is False for v in obs_archive.process_success[miss.name]['emchain'].values()])) and (
                'epchain' in obs_archive.process_success[miss.name] and all(
            [v is False for v in obs_archive.process_success[miss.name]['epchain'].values()])):

            warn("Every emchain and epchain run for the {m} mission in the {a} archive is reporting as a "
                 "failure, skipping process.".format(m=miss.name, a=obs_archive.archive_name), stacklevel=2)
            continue
        else:
            # This fetches those IDs for which emchain has reported success, and these are what we will iterate
            #  through to ensure that we only act upon data that is in a final event list form.
            valid_ids = [k for k, v in obs_archive.process_success[miss.name]['emchain'].items() if v] + \
                        [k for k, v in obs_archive.process_success[miss.name]['epchain'].items() if v]

        # We iterate through the valid IDs rather than nest ObsID and instrument for loops
        for val_id in valid_ids:
            # TODO Review this if I change the IDing system as I was pondering in issue #44
            if 'M1' in val_id:
                obs_id, exp_id = val_id.split('M1')
                inst = 'M1'
            elif 'M2' in val_id:
                obs_id, exp_id = val_id.split('M2')
                # The form of this inst is different to the standard in DAXA/SAS (M2), because emanom log files
                #  are named with mos1 and mos2 rather than M1 and M2
                inst = 'M2'
                evt_list_file = obs_archive.process_extra_info[miss.name]['emchain'][val_id]['evt_list']
                oot_evt_list_file = None
            elif 'PN' in val_id:
                obs_id, exp_id = val_id.split('PN')
                # The form of this inst is different to the standard in DAXA/SAS (M2), because emanom log files
                #  are named with mos1 and mos2 rather than M1 and M2
                inst = 'PN'
                evt_list_file = obs_archive.process_extra_info[miss.name]['epchain'][val_id]['evt_list']
                oot_evt_list_file = obs_archive.process_extra_info[miss.name]['epchain'][val_id]['oot_evt_list']
            else:
                raise ValueError("Somehow there is no instance of M1, M2, or PN in that storage key, this should be "
                                 "impossible!")


    #         # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
    #         #  files will be written to.
    #         dest_dir = obs_archive.get_processed_data_path(miss, obs_id)
    #         ccf_path = dest_dir + 'ccf.cif'
    #
    #         # Set up a temporary directory to work in (probably not really necessary in this case, but will be
    #         #  in other processing functions).
    #         temp_name = "tempdir_{}".format(randint(0, 1e+8))
    #         temp_dir = dest_dir + temp_name + "/"
    #
    #         # Checking for the output anom file created by the process (unless turned off with an argument)
    #         log_name = "{i}{eid}-anom.log".format(i=inst, eid=exp_id)
    #         final_path = dest_dir + log_name
    #
    #         # If it doesn't already exist then we will create commands to generate it
    #         # TODO Decide whether this is the route I really want to follow for this (see issue #28)
    #         if not os.path.exists(final_path):
    #             # Make the temporary directory (it shouldn't already exist but doing this to be safe)
    #             if not os.path.exists(temp_dir):
    #                 os.makedirs(temp_dir)
    #
    #             # Format the blank command string defined near the top of this function with information
    #             #  particular to the current mission and ObsID
    #             cmd = emanom_cmd.format(d=temp_dir, ccf=ccf_path, ef=evt_list_file, of=log_name)
    #
    #             # Now store the bash command, the path, and extra info in the dictionaries
    #             miss_cmds[miss.name][val_id] = cmd
    #             miss_final_paths[miss.name][val_id] = final_path
    #             miss_extras[miss.name][val_id] = {}
    #
    # # This is just used for populating a progress bar during the process run
    # process_message = 'Checking for MOS CCD anomalous states'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress
