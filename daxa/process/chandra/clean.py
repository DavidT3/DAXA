#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 23/10/2024, 12:47. Copyright (c) The Contributors
import os
from random import randint

import numpy as np
from astropy.units import Quantity, UnitConversionError

from daxa import NUM_CORES
from daxa.archive import Archive
from daxa.exceptions import NoDependencyProcessError
from daxa.process.chandra._common import ciao_call, _ciao_process_setup


@ciao_call
def deflare(obs_archive: Archive, method: str = 'sigma', allowed_sigma: float = 3.0, min_length: int = 3,
            time_bin_size: Quantity = Quantity(200, 's'), lc_lo_en: Quantity = Quantity(500, 'eV'),
            lc_hi_en: Quantity = Quantity(7000, 'eV'), num_cores: int = NUM_CORES, disable_progress: bool = False,
            timeout: Quantity = None):
    """
    The DAXA wrapper for the Chandra CIAO task 'deflare', which attempts to identify good time intervals with minimal
    soft-proton flaring. Both ACIS and HRC observations will be processed by this function.

    This function does not generate final event lists, but instead is used to create good-time-interval files
    which are then applied to the creation of final event lists, along with other user-specified filters.

    :param Archive obs_archive: An Archive instance containing a Chandra mission instance. This function will fail
        if no Chandra missions are present in the archive.
    :param str method: The method for the flare-removal tool to use; default is 'sigma', and the other option
        is 'clean'.
    :param float allowed_sigma: For method='sigma', this will control which parts of the lightcurve (anything more than
        sigma standard deviations from the mean); for method='clean' this controls which data are used to calculate
        the mean count-rate. Default is 3.0.
    :param int min_length: The minimum number of consecutive time bins that pass the count-rate filtering performed
        by the 'sigma' method before a good-time-interval (GTI) is declared. Default is 3.
    :param Quantity time_bin_size: Sets the size of the time bin that will be used to generate a light curve for
        the deflaring method to work with. Default is 200 seconds.
    :param Quantity lc_lo_en: The lower energy bound for the light curve used for soft proton flaring
        identification in ACIS data, it will be ignored for HRC due to the limited energy resolution. Default
        is 0.5 keV.
    :param Quantity lc_hi_en: The upper energy bound for the light curve used for soft proton flaring
        identification in ACIS data, it will be ignored for HRC due to the limited energy resolution. Default
        is 7.0 keV.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the CIAO generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire process, but a timeout for individual
        ObsID-Inst processes.
    """
    # Run the setup for Chandra processes, which checks that CIAO is installed (as well as CALDB), and checks that the
    #  archive has at least one Chandra mission in it, and
    ciao_vers, caldb_vers, chan_miss = _ciao_process_setup(obs_archive)

    # We're going to wrap the 'deflare' tool that is included in CIAO, which will make our shiny new GTIs that
    #  exclude periods of intense soft-proton flaring
    # The ACIS command is slightly different from what we allow for HRC, as with ACIS the user can specify an energy
    #  band to generate the light curve within, but HRC has essentially no energy resolution so that isn't
    #  applicable for those data
    acis_df_cmd = ('cd {d}; dmextract infile="{ef}[energy={lo_en}:{hi_en}][bin time=::{bt}]" outfile={lc} opt="ltc1";'
                   'deflare infile={in_f} outfile={out_f} method={me} nsigma={s} minlength={ml} verbose=5; '
                   'cd ..; rm -r {d}')

    hrc_df_cmd = ('cd {d}; dmextract infile="{ef}[bin time=::{bt}]" outfile={lc} opt="ltc1";'
                  'deflare infile={in_f} outfile={out_f} method={me} nsigma={s} minlength={ml} verbose=5; '
                  'cd ..; rm -r {d}')

    # This is the final name of the light-curve we're going to generate to use for the deflaring analysis
    lc_name = "obsid{o}-inst{i}-subexp{se}-lightcurve.fits"

    # This represents the final name of the deflaring GTI
    gti_name = "obsid{o}-inst{i}-subexp{se}-flaringGTI.fits"

    # ---------------------------------- Checking and converting user inputs ----------------------------------
    # Make sure that the method is one of the two allowed options, defined by the two types of cleaning
    #  that the deflare tool can do
    if method not in ['sigma', 'clean']:
        raise ValueError("The 'method' must be either 'sigma', or 'clean'.")
    elif method == 'clean':
        raise NotImplementedError("The lightcurve cleaning method 'lc_clean' is not yet fully supported by "
                                  "DAXA, please contact the developers if you need this functionality.")

    # Now checking the inputs which configure the chosen method
    # The 'min_length' parameter must be an integer, you can't have a float number of time bins
    if not isinstance(min_length, int):
        raise TypeError("The 'min_length' argument must be an integer, as it represents a number of discrete "
                        "time bins.")

    # Have to make sure that the energy bounds are in units that can be converted to eV (which is what the light
    #  curve generation routine expects for these arguments).
    if not lc_lo_en.unit.is_equivalent('eV') or not lc_hi_en.unit.is_equivalent('eV'):
        raise UnitConversionError("The 'lc_lo_en' and 'lc_hi_en' arguments must be astropy quantities in units "
                                  "that can be converted to eV.")
    # Obviously the upper limit can't be lower than the lower limit, or equal to it.
    elif lc_hi_en <= lc_lo_en:
        raise ValueError("The 'lc_hi_en' argument must be larger than the 'lc_lo_en' argument.")
    # Make sure we're converted to the right unit
    else:
        lc_lo_en = lc_lo_en.to('eV').astype(int)
        lc_hi_en = lc_hi_en.to('eV').astype(int)

    # Same deal with the time bin size - it must be in seconds
    if not time_bin_size.unit.is_equivalent('s'):
        raise UnitConversionError("The 'time_bin_size' argument must be an astropy quantity in units that can be "
                                  "converted to seconds.")
    else:
        time_bin_size = time_bin_size.to('s').astype(int)
    # ---------------------------------------------------------------------------------------------------------

    # Sets up storage dictionaries for bash commands, final file paths (to check they exist at the end), and any
    #  extra information
    miss_cmds = {}
    miss_final_paths = {}
    miss_extras = {}

    # We are iterating through Chandra missions, though only one type exists in DAXA and I don't see that changing.
    #  Much of this code is boilerplate that you'll see throughout the Chandra functions (and similar code in many of
    #  the other telescope processing functions), but never mind - it doesn't need to be that different, so why
    #  should we make it so?
    for miss in chan_miss:
        # Sets up the top level keys (mission name) in our storage dictionaries
        miss_cmds[miss.name] = {}
        miss_final_paths[miss.name] = {}
        miss_extras[miss.name] = {}

        # Getting all the ObsIDs that have been flagged as being able to be processed
        all_obs = obs_archive.get_obs_to_process(miss.name)
        # Then filtering those based on which of them successfully passed the dependency function
        good_obs_sel = obs_archive.check_dependence_success(miss.name, all_obs, 'chandra_repro', no_success_error=False)
        good_obs = np.array(all_obs)[good_obs_sel]

        # Have to check that there is something for us to work with here!
        if len(good_obs) == 0:
            raise NoDependencyProcessError("No observations have had successful 'chandra_repro' runs, so "
                                           "deflare cannot be run.")

        for obs_info in good_obs:
            # This is the valid id that allows us to retrieve the specific product for this ObsID-Inst-sub-exposure
            #  (though for Chandra the sub-exposure ID matters very VERY rarely) combo
            val_id = ''.join(obs_info)
            # Split out the information in obs_info
            obs_id, inst, exp_id = obs_info

            # We will need the event list created by the 'chandra_repro' run, so the path must be retrieved
            rel_evt = obs_archive.process_extra_info[miss.name]['chandra_repro'][val_id]['evt_list']

            # This path is guaranteed to exist, as it was set up in _ciao_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.construct_processed_data_path(miss, obs_id)

            # Set up a temporary directory to work in (probably not really necessary in this case, but will be
            #  in other processing functions).
            r_id = randint(0, int(1e+8))
            temp_name = "tempdir_{}".format(r_id)
            temp_dir = dest_dir + temp_name + "/"

            # ------------------------------ Creating final name for output LC & GTI ------------------------------
            # The path for the light-curve - we'll be storing it as well as using it for the deflaring analysis.
            lc_final_path = os.path.join(dest_dir, 'cleaning', lc_name.format(o=obs_id, se=exp_id, i=inst))
            # This is where the final flaring GTI will be stored
            gti_final_path = os.path.join(dest_dir, 'cleaning', gti_name.format(o=obs_id, se=exp_id, i=inst))
            # ----------------------------------------------------------------------------------------------------

            # If it doesn't already exist then we will create commands to generate it
            if ('deflare' not in obs_archive.process_success[miss.name] or
                    val_id not in obs_archive.process_success[miss.name]['deflare']):
                # Make the temporary directory for processing - this (along with the temporary PFILES that
                #  the execute_cmd function will create) should help avoid any file collisions
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                # There are slightly different commands for ACIS and HRC observations - don't allow the user to
                #  set energy bounds for the HRC lightcurve
                if inst == 'ACIS':
                    # Fill out the template, and generate the command that we will run through subprocess
                    cmd = acis_df_cmd.format(d=temp_dir, ef=rel_evt, lo_en=lc_lo_en.value, hi_en=lc_hi_en.value,
                                             bt=time_bin_size.value, lc=lc_final_path, in_f=lc_final_path,
                                             out_f=gti_final_path, me=method, s=allowed_sigma, ml=min_length)
                else:
                    cmd = hrc_df_cmd.format(d=temp_dir, ef=rel_evt, bt=time_bin_size.value, lc=lc_final_path,
                                            in_f=lc_final_path, out_f=gti_final_path, me=method, s=allowed_sigma,
                                            ml=min_length)

                # Now store the bash command, the path, and extra info in the dictionaries
                miss_cmds[miss.name][val_id] = cmd
                miss_final_paths[miss.name][val_id] = gti_final_path
                miss_extras[miss.name][val_id] = {'working_dir': temp_dir, 'flaring_gti': gti_final_path,
                                                  'lightcurve': lc_final_path}

    # This is just used for populating a progress bar during the process run
    process_message = 'Finding flares in observations'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout
