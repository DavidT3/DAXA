#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 16/01/2023, 18:25. Copyright (c) The Contributors
import os
from random import randint
from typing import Union, Tuple
from warnings import warn

from astropy.units import Quantity, UnitConversionError

from daxa import NUM_CORES
from daxa.archive.base import Archive
from daxa.exceptions import NoDependencyProcessError
from daxa.process.xmm._common import _sas_process_setup, ALLOWED_XMM_MISSIONS, sas_call


@sas_call
def espfilt(obs_archive: Archive, method: str = 'histogram', with_smoothing: Union[bool, Quantity] = True,
            with_binning: Union[bool, Quantity] = True, ratio: float = 1.2, lo_en: Quantity = Quantity(2500, 'eV'),
            hi_en: Quantity = Quantity(8500, 'eV'), range_scale: dict = None, allowed_sigma: float = 3.0,
            gauss_fit_lims: Tuple[float, float] = (0.1, 6.5), num_cores: int = NUM_CORES,
            disable_progress: bool = False):
    """
    The DAXA wrapper for the XMM SAS task espfilt, which attempts to identify good time intervals with minimal
    soft-proton flaring for individual sub-exposures (if multiple have been taken) of XMM ObsID-Instrument
    combinations. Both EPIC-PN and EPIC-MOS observations will be processed by this function.

    :param Archive obs_archive: An Archive instance containing XMM mission instances with PN/MOS observations for
        which espfilt should be run. This function will fail if no XMM missions are present in the archive.
    :param str method: The method that espfilt should use to find soft proton flaring. Either 'ratio' or 'histogram'
        can be selected. The default is 'histogram'.
    :param bool/Quantity with_smoothing: Should smoothing be applied to the light curve data. If set to True (the
        default) a smoothing factor of 51 seconds is used, if set to False smoothing will be turned off, if an astropy
        Quantity is passed (with units convertible to seconds) then that value will be used for the smoothing factor.
    :param bool/Quantity with_binning: Should binning be applied to the light curve data. If set to True (the
        default) a bin size of 60 seconds is used, if set to False binning will be turned off, if an astropy
        Quantity is passed (with units convertible to seconds) then that value will be used for the bin size.
    :param float ratio: Flaring ratio of annulus counts.
    :param Quantity lo_en: The lower energy bound for the event lists used for soft proton flaring identification.
    :param Quantity hi_en: The upper energy bound for the event lists used for soft proton flaring identification.
    :param dict range_scale: Histogram fit range scale factor. The default is a dictionary with an entry for 'pn'
        (15.0) and an entry for 'mos' (6.0).
    :param float allowed_sigma: Limit in sigma for unflared rates.
    :param Tuple[float, float] gauss_fit_lims: The parameter limits for gaussian fits.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :return: Information required by the SAS decorator that will run commands. Top level keys of any dictionaries are
        internal DAXA mission names, next level keys are ObsIDs. The return is a tuple containing a) a dictionary of
        bash commands, b) a dictionary of final output paths to check, c) a dictionary of extra info (in this case
        obs and analysis dates), d) a generation message for the progress bar, e) the number of cores allowed, and
        f) whether the progress bar should be hidden or not.
    :rtype: Tuple[dict, dict, dict, str, int, bool]
    """
    # Run the setup for SAS processes, which checks that SAS is installed, checks that the archive has at least
    #  one XMM mission in it, and shows a warning if the XMM missions have already been processed
    sas_version = _sas_process_setup(obs_archive)

    # Checking to make sure an acceptable value for 'method' has been passed
    if method != 'histogram' and method != 'ratio':
        raise ValueError("The string passed for 'method' must be either histogram or ratio.")

    # Parsing the user's choice of parameters for espfilt smoothing
    if not isinstance(with_smoothing, (bool, Quantity)):
        raise TypeError("The with_smoothing parameter must be either boolean or an astropy quantity.")
    # If with_smoothing is boolean then no custom value for smooth_factor has been passed, so we use the
    #  espfilt default
    elif isinstance(with_smoothing, bool):
        smooth_factor = Quantity(51, 's')
    # If with_smoothing wasn't boolean then it must be a Quantity - so must check the units
    elif not with_smoothing.unit.is_equivalent('s'):
        raise UnitConversionError("If the 'with_smoothing' argument is an Astropy Quantity, it must be in units that"
                                  " can be converted to seconds.")
    # Ensure that the value is within the allowed limits specified in the espfilt documentation
    elif with_smoothing < Quantity(1, 's') or with_smoothing > Quantity(60, 's'):
        raise ValueError("If the with_smoothing argument is set (to activate light-curve smoothing) then it must be "
                         "set to a value greater than 1 second and less than 60 seconds.")
    elif isinstance(with_smoothing, Quantity):
        smooth_factor = with_smoothing.to('s')
        # If a custom smoothing factor has been passed then the smoothing will be turned on
        with_smoothing = True

    # Parsing the user's choice of parameters for espfilt binning
    if not isinstance(with_binning, (bool, Quantity)):
        raise TypeError("The with_binning parameter must be either boolean or an astropy quantity.")
    # If with_binning is boolean then no custom value for bin_size has been passed, so we use the espfilt default
    elif isinstance(with_binning, bool):
        bin_size = Quantity(60, 's')
    # If with_binning wasn't boolean then it must be a Quantity - so must check the units
    elif not with_binning.unit.is_equivalent('s'):
        raise UnitConversionError("If the 'with_binning' argument is an Astropy Quantity, it must be in units that"
                                  " can be converted to seconds.")
    # Ensure that the value is within the allowed limits specified in the espfilt documentation
    elif with_binning < Quantity(1, 's'):
        raise ValueError("If the with_binning argument is set (to activate light-curve binning) then it must be set "
                         "to a value greater than 1 second.")
    elif isinstance(with_binning, Quantity):
        bin_size = with_binning.to('s')
        # If a custom bin size has been passed then the binning will be turned on
        with_binning = True

    # The only check we make on the flaring ratio of annulus counts is to ensure it's an integer/float - no limits
    #  were specified in the parameter docs of espfilt
    if not isinstance(ratio, (int, float)):
        raise TypeError("The ratio argument must be either a float or an integer.")

    # Have to make sure that the energy bounds are in units that can be converted to eV (which is what espfilt
    #  expects for these arguments).
    if not lo_en.unit.is_equivalent('eV') or not hi_en.unit.is_equivalent('eV'):
        raise UnitConversionError("The lo_en and hi_en arguments must be astropy quantities in units that can be "
                                  "converted to eV.")
    # Obviously the upper limit can't be lower than the lower limit, or equal to it.
    elif hi_en <= lo_en:
        raise ValueError("The hi_en argument must be larger than the lo_en argument.")
    # Make sure we're converted to the right unit
    else:
        lo_en = lo_en.to('eV')
        hi_en = hi_en.to('eV')

    # Also enforce the value limits specified in the espfilt documentation
    if (lo_en < Quantity(1, 'eV') or lo_en > Quantity(32766, 'eV')) or \
            (hi_en < Quantity(2, 'eV') or hi_en > Quantity(32767, 'eV')):
        raise ValueError("The lo_en value must be greater than 1 eV and less than 32766 eV, the hi_en value must be "
                         "greater than 2 eV and lower than 32767 eV.")

    # Setting the default values for range scale, can't have a mutable argument
    if range_scale is None:
        # These are the default values from the parameter docs of espfilt, for MOS and PN
        range_scale = {'mos': 6.0, 'pn': 15.0}
    # Have to make sure that if the user isn't using the defaults, they have passed the data in the form that we
    #  expect it to be - don't want to pass dodgy parameters to espfilt as it is better for Python exceptions to
    #  be called than bash errors to pop up.
    elif range_scale is not None and not isinstance(range_scale, dict):
        raise ValueError("The range_scale argument must be a dictionary with an entry for MOS instruments (with key "
                         "'mos') and an entry for the PN instrument (with key 'pn').")
    # Make sure that there is an entry for MOS and PN if the user has passed their own
    elif range_scale is not None and isinstance(range_scale, dict) and \
            ('pn' not in range_scale or 'mos' not in range_scale):
        raise KeyError("The range_scale argument must be a dictionary with an entry for MOS instruments (with key "
                       "'mos') and an entry for the PN instrument (with key 'pn').")

    # The only check we make on the limit in sigma for unflared rates is to ensure it's an integer/float - no limits
    #  were specified in the parameter docs of espfilt
    if not isinstance(allowed_sigma, (int, float)):
        raise TypeError("The allowed_sigma argument must be either an integer or a float.")

    # This should be a tuple with two int/float entries, these checks make sure of that
    if not isinstance(gauss_fit_lims, tuple):
        raise TypeError("The gauss_fit_lims argument must be a tuple.")
    elif len(gauss_fit_lims) != 2:
        raise ValueError("The gauss_fit_lims tuple must have two elements; the first the lower limit, and the second "
                         "the upper limit.")
    elif not all([isinstance(g_lim, (float, int)) for g_lim in gauss_fit_lims]):
        raise TypeError("The elements of gauss_fit_lims must be either integers or floats.")
    # The upper limit cannot be less or equal to the lower limit, it wouldn't make sense
    elif gauss_fit_lims[1] <= gauss_fit_lims[0]:
        raise ValueError("The second (upper) entry in the gauss_fit_lims argument cannot be less than or equal to"
                         " the first (lower) entry.")
    # Finally, we inflict the value limits specified in the espfilt parameter docs
    elif any([g_lim < 0 or g_lim > 10 for g_lim in gauss_fit_lims]):
        raise ValueError("The entries in gauss_fit_lims must be greater than zero, and less than 10.")

    # Define the form of the espfilt command to clean the event lists for soft protons, then copy the GTI file, the
    #  cleaned events list within the energy bands, and the diagnostic histogram
    ef_cmd = "cd {d}; export SAS_CCF={ccf}; espfilt eventfile={ef} withoot={woot} ootfile={oot} method={me} " \
             "withsmoothing={ws} smooth={s} withbinning={wb} binsize={bs} ratio={r} withlongnames=yes elow={el} " \
             "ehigh={eh} rangescale={rs} allowsigma={asi} keepinterfiles=no limits={gls}; mv {gti} ../; " \
             "mv {allev} ../; mv {hist} ../; cd ../; rm -r {d}"

    # Need to change parameter to turn on smoothing if the user wants it. The parameter
    #  must be changed from boolean to a 'yes' or 'no' string because that is what espfilt wants
    if with_smoothing:
        with_smoothing = 'yes'
    else:
        with_smoothing = 'no'
    # Can't pass an astropy quantity as its string representation will contain a unit, we need to just
    #  extract the value - which we have made sure is in the correct units
    smooth_factor = int(smooth_factor.value)

    # Also need to change a parameter to turn on binning if the user wants it. The parameter
    #  must be changed from boolean to a 'yes' or 'no' string because that is what espfilt wants
    if with_binning:
        with_binning = 'yes'
    else:
        with_binning = 'no'
    # Can't pass an astropy quantity as its string representation will contain a unit, we need to just
    #  extract the value - which we have made sure is in the correct units
    bin_size = int(bin_size.value)

    # Make sure the energy limits are an integer, and that they aren't an astropy quantity
    lo_en = int(lo_en.value)
    hi_en = int(hi_en.value)

    # Finally, the tuple of lower and upper gaussian fit limits need to be a string representation. Apparently
    #  needs to be space separated and in quotation marks for the call to work
    gauss_fit_lims = '"' + " ".join([str(gl) for gl in gauss_fit_lims]) + '"'

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
        #  it is a requirement for this processing function. There will probably be a more elegant way of checking
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
                alt_inst = 'mos1'
                evt_list_file = obs_archive.process_extra_info[miss.name]['emchain'][val_id]['evt_list']
                oot_evt_list_file = 'dataset'
            elif 'M2' in val_id:
                obs_id, exp_id = val_id.split('M2')
                # The form of this inst is different to the standard in DAXA/SAS (M2), because emanom log files
                #  are named with mos1 and mos2 rather than M1 and M2
                inst = 'M2'
                alt_inst = 'mos2'
                evt_list_file = obs_archive.process_extra_info[miss.name]['emchain'][val_id]['evt_list']
                oot_evt_list_file = 'dataset'
            elif 'PN' in val_id:
                obs_id, exp_id = val_id.split('PN')
                # The form of this inst is different to the standard in DAXA/SAS (M2), because emanom log files
                #  are named with mos1 and mos2 rather than M1 and M2
                inst = 'PN'
                alt_inst = 'pn'
                evt_list_file = obs_archive.process_extra_info[miss.name]['epchain'][val_id]['evt_list']
                oot_evt_list_file = obs_archive.process_extra_info[miss.name]['epchain'][val_id]['oot_evt_list']
            else:
                raise ValueError("Somehow there is no instance of M1, M2, or PN in that storage key, this should be "
                                 "impossible!")

            # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.get_processed_data_path(miss, obs_id)
            ccf_path = dest_dir + 'ccf.cif'

            # Set up a temporary directory to work in (probably not really necessary in this case, but will be
            #  in other processing functions).
            temp_name = "tempdir_{}".format(randint(0, 1e+8))
            temp_dir = dest_dir + temp_name + "/"

            # Setting up the paths to the event file, GTI file, and diagnostic histogram - these will be checked
            #  for at the end to ensure that the process worked. Need to use 'alt_inst' here because the files
            #  produced have mos1 rather than M1, mos2 rather than M2 in their names
            evt_name = "{i}{exp_id}-allevc-{l}-{u}.fits".format(i=alt_inst, exp_id=exp_id, l=lo_en, u=hi_en)
            gti_name = "{i}{exp_id}-gti-{l}-{u}.fits".format(i=alt_inst, exp_id=exp_id, l=lo_en, u=hi_en)
            hist_name = "{i}{exp_id}-hist-{l}-{u}.qdp".format(i=alt_inst, exp_id=exp_id, l=lo_en, u=hi_en)
            evt_path = dest_dir + evt_name
            gti_path = dest_dir + gti_name
            hist_path = dest_dir + hist_name
            final_paths = [evt_path, gti_path, hist_path]

            # If it doesn't already exist then we will create commands to generate it
            # TODO Need to decide which file to check for here to see whether the command has already been run
            # Make the temporary directory (it shouldn't already exist but doing this to be safe)
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # Format the blank command string defined near the top of this function - in this case the
            #  configuration needs to change depending on the instrument and user configuration
            if inst == 'PN':
                with_oot = 'yes'
                rs = range_scale['pn']
            else:
                with_oot = 'no'
                rs = range_scale['mos']

            cmd = ef_cmd.format(d=temp_dir, ccf=ccf_path, ef=evt_list_file, woot=with_oot, oot=oot_evt_list_file,
                                me=method, ws=with_smoothing, s=smooth_factor, wb=with_binning, bs=bin_size,
                                r=ratio, el=lo_en, eh=hi_en, rs=rs, asi=allowed_sigma, gls=gauss_fit_lims,
                                gti=gti_name, hist=hist_name, allev=evt_name)

            # Now store the bash command, the path, and extra info in the dictionaries
            miss_cmds[miss.name][val_id] = cmd
            miss_final_paths[miss.name][val_id] = final_paths
            miss_extras[miss.name][val_id] = {}

    # This is just used for populating a progress bar during the process run
    process_message = 'Finding PN/MOS soft-proton flares'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress
