#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 03/09/2024, 14:52. Copyright (c) The Contributors

import os.path
from random import randint

from astropy.units import Quantity, UnitConversionError

from daxa import NUM_CORES
from daxa.archive.base import Archive
from daxa.exceptions import NoDependencyProcessError
from daxa.process._cleanup import _last_process
from daxa.process.erosita._common import _esass_process_setup, ALLOWED_EROSITA_MISSIONS, esass_call


@_last_process(ALLOWED_EROSITA_MISSIONS, 1)
@esass_call
def cleaned_evt_lists(obs_archive: Archive, lo_en: Quantity = Quantity(0.2, 'keV'),
                      hi_en: Quantity = Quantity(10, 'keV'), flag: int = 0xc0000000, flag_invert: bool = True,
                      pattern: int = 15, num_cores: int = NUM_CORES, disable_progress: bool = False,
                      timeout: Quantity = None):
    """
    The function wraps the eROSITA eSASS task evtool, which is used for selecting events.
    This has been tested up to evtool v2.10.1

    This function is used to apply the soft-proton filtering (along with any other filtering you may desire, including
    the setting of energy limits) to eROSITA event lists, resulting in the creation of sets of cleaned event lists
    which are ready to be analysed.

    :param Archive obs_archive: An Archive instance containing eROSITA mission instances with observations for
        which cleaned event lists should be created. This function will fail if no eROSITA missions are present in
        the archive.
    :param Quantity lo_en: The lower bound of an energy filter to be applied to the cleaned, filtered, event lists. If
        'lo_en' is set to an Astropy Quantity, then 'hi_en' must be as well. Default is 0.2 keV, which is the
        minimum allowed by the eROSITA toolset. Passing None will result in the default value being used.
    :param Quantity hi_en: The upper bound of an energy filter to be applied to the cleaned, filtered, event lists. If
        'hi_en' is set to an Astropy Quantity, then 'lo_en' must be as well. Default is 10 keV, which is the
        maximum allowed by the eROSITA toolset. Passing None will result in the default value being used.
    :param int flag: FLAG parameter to select events based on owner, information, rejection, quality, and corrupted
        data. The eROSITA website contains the full description of event flags in section 1.1.2 of the following link:
        https://erosita.mpe.mpg.de/edr/DataAnalysis/prod_descript/EventFiles_edr.html. The default parameter will
        select all events flagged as either singly corrupt or as part of a corrupt frame.
    :param bool flag_invert: If set to True, this function will discard all events selected by the flag parameter.
        This is the default behaviour.
    :param int pattern: Selects events of a certain pattern chosen by the integer key. The default of 15 selects
        all four of the recognized legal patterns.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the eSASS generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire cleaned_evt_lists process, but a timeout for individual
        ObsID-Inst-subexposure processes.
    """
    # Run the setup for eSASS processes, which checks that eSASS is installed, checks that the archive has at least
    #  one eROSITA mission in it, and shows a warning if the eROSITA missions have already been processed
    esass_in_docker = _esass_process_setup(obs_archive)

    # We ensure that if a null value is passed the lo_en and hi_en values revert to default behaviour
    if lo_en is None:
        lo_en = Quantity(0.2, 'keV')
    if hi_en is None:
        hi_en = Quantity(10.0, 'keV')

    # Checking user's choice of energy limit parameters
    if not isinstance(lo_en, Quantity) or not isinstance(hi_en, Quantity):
        raise TypeError("The lo_en and hi_en arguments must be astropy quantities in units "
                        "that can be converted to keV.")
    
    # Have to make sure that the energy bounds are in units that can be converted to keV (which is what evtool
    #  expects for these arguments).
    elif not lo_en.unit.is_equivalent('eV') or not hi_en.unit.is_equivalent('eV'):
        raise UnitConversionError("The lo_en and hi_en arguments must be astropy quantities in units "
                                  "that can be converted to keV.")

    # Checking that the upper energy limit is not below the lower energy limit
    elif hi_en <= lo_en:
        raise ValueError("The hi_en argument must be larger than the lo_en argument.")
    
    # Converting to the right unit
    else:
        lo_en = lo_en.to('keV')
        hi_en = hi_en.to('keV')

    # Checking user's lo_en and hi_en inputs are in the valid energy range for eROSITA
    if (lo_en < Quantity(200, 'eV') or lo_en > Quantity(10000, 'eV')) or \
       (hi_en < Quantity(200, 'eV') or hi_en > Quantity(10000, 'eV')):
        raise ValueError("The lo_en and hi_en value must be between 0.2 keV and 10 keV.")

    # The eSASS software has a bug when the user specifies the flag inversion parameter
    # so for the moment we wont let the user chose the flag
    if flag != 0xc0000000:
        raise NotImplementedError("DAXA currently doesn't support flag selection due to a bug "
                                  "within the eSASS software.")
    # Checking user has input the flag parameter as an integer
    #if not isinstance(flag, int):
    #    raise TypeError("The flag parameter must be an integer.")

    # Checking the input is a valid hexidecimal number
    #if not _is_valid_flag(flag):
    #    raise ValueError("{} is not a valid eSASS flag, see the eROSITA website"
    #                    " for valid flags.".format(flag))
    
    if not flag_invert:
        raise NotImplementedError("DAXA currently doesn't support flag selection due to a bug "
                                  "within the eSASS software.")
        
    # Checking user has input flag_invert as a boolean
    if not isinstance(flag_invert, bool):
        raise TypeError("The flag_invert parameter must be a boolean.")
    
    # Checking user has input pattern as an integer
    if not isinstance(pattern, int):
        raise TypeError("The pattern parameter must be an integer between 1 and 15 inclusive.")
    
    # Checking user has input a valid pattern
    if pattern <= 0 or pattern >= 16:
        raise ValueError("Valid eROSITA patterns are between 1 and 15 inclusive")
    
    # Converting the parameters to the correct format for the esass command
    lo_en = lo_en.value
    hi_en = hi_en.value

    #if flag_invert:
    #    flag_invert = 'yes'
    #else:
    #    flag_invert = 'no'

    # Define the form of the evtool command that must be run for event list filtering to take place
    evtool_cmd = "cd {d}; evtool eventfiles={ef} gti=FLAREGTI outfile={of} pattern={p} " \
                 "emin={emin} emax={emax}; mv {of} {fep}; rm -r {d}"
    #evtool_cmd = "cd {d}; evtool eventfiles={ef} gti=FLAREGTI outfile={of} pattern={p} " \
    #             " flag={f} flag_invert={fi} emin={emin} emax={emax}; mv {of} {fep}; rm -r {d}"

    # Sets up storage dictionaries for bash commands, final file paths (to check they exist at the end), and any
    #  extra information that might be useful to provide to the next step in the generation process
    miss_cmds = {}
    miss_final_paths = {}
    miss_extras = {}

    # Just grabs the eROSITA missions, we already know there will be at least one because otherwise
    #  _esass_process_setup would have thrown an error
    erosita_miss = [mission for mission in obs_archive if mission.name in ALLOWED_EROSITA_MISSIONS]
    # We are iterating through erosita missions (options could include erosita_cal_pv for instance).
    for miss in erosita_miss:
        # Sets up the top level keys (mission name) in our storage dictionaries
        miss_cmds[miss.name] = {}
        miss_final_paths[miss.name] = {}
        miss_extras[miss.name] = {}

        # This method will fetch the valid data (ObsID, Instruments) that can be processed
        all_obs_info = obs_archive.get_obs_to_process(miss.name)
            
        # Checking that any valid observations are left after the get_obs_to_process function is run
        if len(all_obs_info) == 0:
            raise FileNotFoundError("No valid observations have been found, so cleaned_evt_lists may not be run.")

        # all_obs_info is a list of lists, where each list is of the format: [ObsID, Inst, 'usable'].
        # There is a new list for each instrument, but I just want to loop over the ObsID in the following
        #  bit of code,
        # I also want to know all the instruments that the ObsID contains events for
        # So here I am just making a dictionary of the format: {ObsID: insts}
        # Getting unique obs_ids in all_obs_info
        obs_ids = list(set([all_obs_info_list[0] for all_obs_info_list in all_obs_info]))
        obs_info_dict = {}
        for obs in obs_ids:
            # Collecting all the insts that a certain ObsID has events for
            insts = ''.join([all_obs_info_list[1] for all_obs_info_list in all_obs_info if obs in all_obs_info_list])
            # The insts are all TM{x} where x is a number from 1-7, we want to separate them with _ for the file names
            obs_info_dict[obs] = '_'.join("TM"+ch for ch in insts if ch.isdigit())

        # Counter for number of ObsIDs that flaregti has not been run successfully on
        bad_obs_counter = 0
        # We iterate through the valid identifying information
        for obs_id in obs_info_dict:
            try:
                # Checking that flaregti has been run successfully on this observation so that it can be cleaned
                # Then only writing a command for ObsIDs that have had flaregti successfully run on them
                obs_archive.check_dependence_success(miss.name, obs_id, 'flaregti')
                
                # Getting the insts associated with this obs for file naming purposes 
                insts = obs_info_dict[obs_id]

                # Search through the process_extra_info attribute of the archive to find the paths 
                #   to the event lists
                evt_list_file = obs_archive._process_extra_info[miss.name][obs_id]['path']

                # This path is guaranteed to exist, as it was set up in _esass_process_setup. This is where output
                #  files will be written to.
                dest_dir = obs_archive.construct_processed_data_path(miss, obs_id)
                # Set up a temporary directory to work in (probably not really necessary in this case, but will be
                #  in other processing functions).
                temp_name = "tempdir_{}".format(randint(0, int(1e+8)))
                temp_dir = dest_dir + temp_name + "/"

                # Setting the paths to the output cleaned event list file
                filt_evt_name = "obsid{o}-inst{i}-subexpALL-en{l}_{u}keV-finalevents.fits".format(i=insts, l=lo_en,
                                                                                                  u=hi_en, o=obs_id)
                filt_evt_path = os.path.join(dest_dir, 'events', filt_evt_name)

                # The path that needs to exist is the filtered event list 
                final_paths = [filt_evt_path]

                if ('cleaned_evt_lists' not in obs_archive.process_success[miss.name] or
                        obs_id not in obs_archive.process_success[miss.name]['cleaned_evt_lists']):
                    # Make the temporary directory (it shouldn't already exist but doing this to be safe)
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)

                    cmd = evtool_cmd.format(d=temp_dir, ef=evt_list_file, of=filt_evt_name, f=flag, fi=flag_invert,
                                            p=pattern, emin=lo_en, emax=hi_en, fep=filt_evt_path)

                    # Now store the bash command, the path, and extra info in the dictionaries
                    miss_cmds[miss.name][obs_id] = cmd
                    miss_final_paths[miss.name][obs_id] = final_paths
                    miss_extras[miss.name][obs_id] = {'final_evt': filt_evt_path, 'flag': flag,
                                                      'flag_invert': flag_invert, 'pattern': pattern}
                
            except NoDependencyProcessError:
                # If archive.check_dependence_success raises this error, it means flaregti was not run
                # successfully, and so a warning will be raised saying this observation has not been cleaned
                bad_obs_counter += 1
                pass

        # TODO THIS SHOULD BE REMOVED WHEN I'VE MADE SURE THE DEPENDENCY CHECKER WORKS FOR EROSITA
        # If no observations have had flaregti run successfully, then no events can be cleaned
        if bad_obs_counter == len(obs_info_dict):
            raise NoDependencyProcessError("The required process flaregti has not been run successfully "
                                           "for any data in {mn}".format(mn=miss.name))

    # This is just used for populating a progress bar during the process run
    process_message = 'Generating final event lists'

    return (miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout,
            esass_in_docker)
