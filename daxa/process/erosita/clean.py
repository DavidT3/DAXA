# This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
# Last modified by David J Turner (turne540@msu.edu) Thu Apr 20 2023, 10:52. Copyright (c) The Contributors
import os
from random import randint
from typing import Union

from astropy.units import Quantity, UnitConversionError, def_unit, add_enabled_units, ct, deg, s

from daxa import NUM_CORES
from daxa.archive.base import Archive
from daxa.exceptions import NoDependencyProcessError
from daxa.process.erosita._common import _esass_process_setup, ALLOWED_EROSITA_MISSIONS

# JESS_TODO write an esass call wrapper 
# JESS_TODO check each argument type is correct
# JESS_TODO put arguments as quantities, need to do .value when putting it into cmd line
# JESS_TODO see what the limits are on xmax, xmin is it the size of a sweep
# JESS_TODO see how it deals with sweeps vs. pointing
# DAVID_QUESTION not sure how to deal with skypixel

# DAVID_QUESTION not sure where i should put this bit of code?
# defining surface brightness rate astropy unit for use in flaregti to measure thresholds in 
sb_rate = def_unit('sb_rate', ct / (deg**2 *s)) 
# adding this to enabled units so that it can be used in flaregti
add_enabled_units([sb_rate])

def flaregti(obs_archive: Archive, pimin: Quantity = Quantity(200, 'eV'), pimax: Quantity = Quantity(10000, 'eV'), mask_pimin: Quantity = (200, 'eV'), 
            mask_pimax: Quantity = Quantity(10000, 'eV'), binsize: int = 1200, detml: Union[float, int] = 10, timebin: Quantity = Quantity(20, 's'), 
            source_size: Quantity = Quantity(25, 'arcsec'), source_like: Union[float, int] = 10, threshold: Quantity = Quantity(-1, 'sb_rate'), 
            max_threshold: Quantity = Quantity(-1, 'sb_rate'), mask_iter: int = 3, num_cores: int = NUM_CORES, disable_progress: bool = False, timeout: Quantity = None):
    """
    The DAXA wrapper for the eROSITA eSASS task flaregti, which attempts to identify good time intervals with minimal flaring.
    This has been tested up to flaregti v1.20.

    This function does not generate final event lists, but instead is used to create good-time-interval files
    which are then applied to the creation of final event lists, along with other user-specified filters, in the
    'cleaned_evt_lists' function.

    :param obs_archive Archive: An Archive instance containing eROSITA mission instances with observations for
        which flaregti should be run. This function will fail if no eROSITA missions are present in the archive.
    :param float pimin:  Lower PI bound of energy range for lightcurve creation.
    :param float pimax:  Upper PI bound of energy range for lightcurve creation.
    :param float mask_pimin: Lower PI bound of energy range for finding sources to mask.
    :param float mask_pimax: Upper PI bound of energy range for finding sources to mask.
    :param int binsize: Bin size of mask image (unit: sky pixels).
    :param int detml: Likelihood threshold for mask creation.
    :param int timebin: Bin size for lightcurve (unit: seconds).
    :param int source_size: Diameter of source extracton area for dynamic threshold calculation (unit: arcsec);
        this is the most important parameter if optimizing for extended sources.
    :param int source_like: Source likelihood for automatic threshold calculation.
    :param float threshold: Flare threshold; dynamic if negative (unit: counts/deg^2/sec).
    :param float max_threshold: Maximum threshold rate, if positive (unit: counts/deg^2/sec),
        if set this forces the threshold to be this rate or less.
    :param int mask_iter: Number of repetitions of source masking and GTI creation.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire flaregti process, but a timeout for individual
        ObsID-Inst-subexposure processes.
    :return: Information required by the eSASS decorator that will run commands. Top level keys of any dictionaries are
        internal DAXA mission names, next level keys are ObsIDs. The return is a tuple containing a) a dictionary of
        bash commands, b) a dictionary of final output paths to check, c) a dictionary of extra info (in this case
        obs and analysis dates), d) a generation message for the progress bar, e) the number of cores allowed, and
        f) whether the progress bar should be hidden or not.
    :rtype: Tuple[dict, dict, dict, str, int, bool, Quantity]
    """

    # Run the setup for eSASS processes, which checks that eSASS is installed, checks that the archive has at least
    #  one eROSITA mission in it, and shows a warning if the eROSITA missions have already been processed
    esass_in_docker = _esass_process_setup(obs_archive)

    # Checking user's choice of energy limit parameters
    if not isinstance(pimin, Quantity) or not isinstance(pimax, Quantity):
        raise TypeError("The pimin and pimax arguments must be astropy quantities in units "
                        "that can be converted to eV.")
    
    # Have to make sure that the energy bounds are in units that can be converted to eV (which is what flaregti
    #  expects for these arguments).
    elif not pimin.unit.is_equivalent('eV') or not pimax.unit.is_equivalent('eV'):
        raise UnitConversionError("The pimin and pimax arguments must be astropy quantities in units "
                                  "that can be converted to eV.")

    # Checking that the upper energy limit is not below the lower energy limit (for the lightcurve)
    elif pimax <= pimin:
        raise ValueError("The pimax argument must be larger than the pimin argument.")
    
    # Converting to the right unit
    else:
        pimin = pimin.to('eV')
        pimax = pimax.to('eV')

    # Checking user's pimin and pimax inputs are in the valid energy range for eROSITA
    if (pimin < Quantity(200, 'eV') or pimin > Quantity(10000, 'eV')) or \
        (pimax < Quantity(200, 'eV') or pimax > Quantity(10000, 'eV')):
        raise ValueError("The pimin and pimax value must be between 200 eV and 10000 eV.")
    
    # Repeating these checks but for the image energy range limits
    if not isinstance(mask_pimin, Quantity) or not isinstance(mask_pimax, Quantity):
        raise TypeError("The mask_pimin and mask_pimax arguments must be astropy quantities in units "
                        "that can be converted to eV.")
    
    # Have to make sure that the energy bounds are in units that can be converted to eV (which is what flaregti
    #  expects for these arguments).
    elif not mask_pimin.unit.is_equivalent('eV') or not mask_pimax.unit.is_equivalent('eV'):
        raise UnitConversionError("The mask_pimin and mask_pimax arguments must be astropy quantities in units "
                                  "that can be converted to eV.")
    
    # Checking that the upper energy limit is not below the lower energy limit (for the image)
    elif mask_pimax <= mask_pimin:
        raise ValueError("The mask_pimax argument must be larger than the mask_pimin argument.")

    # Converting to the right unit
    else:
        mask_pimin = mask_pimin.to('eV')
        mask_pimax = mask_pimax.to('eV')

    # Checking user's mask_pimin and mask_pimax inputs are in the valid energy range for eROSITA
    if (mask_pimin < Quantity(200, 'ev') or mask_pimin > Quantity(10000, 'eV')) or \
       (mask_pimax < Quantity(200, 'eV') or mask_pimax > Quantity(10000, 'eV')):
        raise ValueError("The mask_pimin and mask_pimax value must be between 200 eV and 10000 eV.")
    
    # Checking user's choice for the timebin parameter
    if not isinstance(timebin, Quantity):
        raise TypeError("The timebin argument must be an astropy quantity in units "
                        "that can be converted to seconds.")

    # Have to make sure that the timebin is in units that can be converted to s (which is what flaregti
    #  expects for this argument).
    elif not timebin.unit.is_equivalent('s'):
        raise UnitConversionError("The timebin argument must be an astropy quantity in units "
                                  "that can be converted to seconds.")

    # Converting to the right unit                              
    else:
        timebin = timebin.to('s')

    # Avoiding the operating system error you get when you enter too large of a timebin into flaregti
    if timebin > 1000000000:
        raise ValueError("Please enter a timebin argument equivalent to less than 1000000000s.")
    
    # Not allowing a negative timebin to be entered
    if timebin <= 0:
        raise ValueError("The timebin argument may not be negative or equal to 0.")
    
    # Checking user's choice for the source_size parameter
    if not isinstance(source_size, Quantity):
        raise TypeError("The source_size argument must be an astropy quantity in units "
                        "that can be converted to arcseconds.")

    # Have to make sure that the timebin is in units that can be converted to s (which is what flaregti
    #  expects for this argument).
    elif not source_size.unit.is_equivalent('arcsec'):
        raise UnitConversionError("The source_size argument must be an astropy quantity in units "
                                  "that can be converted to arcseconds.")

    # Converting to the right unit                              
    else:
        source_size = source_size.to('arcsec')

    # Checking user's choice for the binsize parameter
    if not isinstance(binsize, int):
        raise TypeError("The binsize argument must be an integer.")

    # Checking the validity of the binsize value
    elif binsize <= 0:
        raise ValueError("The binsize argument may not be negative or equal to 0.")
    
    # Checking user's choice for the detml parameter
    if not isinstance(detml, (int, float)):
        raise TypeError("The detml argument must be an integer or a float.")
    
    # Checking user's choice for the source_like parameter
    if not isinstance(source_like, (int, float)):
        raise TypeError("The source_like argument must be an integer or a float.")
    
    # Checking user's choice for the threshold parameter
    if not isinstance(threshold, Quantity):
        raise TypeError("The threshold argument must be an astropy quantity in units that can "
                        "be converted into counts/deg^2/s.")

    # Checking it is in the correct units
    elif not threshold.is_equivalent('sb_rate'):
        raise UnitConversionError("The threshold argument must be an astropy quantity in units that can "
                                  "be converted into counts/deg^2/s.")
    
    # Converting to the right unit                              
    else:
        threshold = threshold.to('sb_rate')
    
    # Checking user's choice for the max_threshold parameter
    if not isinstance(max_threshold, Quantity):
        raise TypeError("The max_threshold argument must be an astropy quantity in units that can "
                        "be converted into counts/deg^2/s.")

    # Checking it is in the correct units
    elif not max_threshold.is_equivalent('sb_rate'):
        raise UnitConversionError("The max_threshold argument must be an astropy quantity in units that can "
                                  "be converted into counts/deg^2/s.")
    
    # Converting to the right unit                              
    else:
        max_threshold = max_threshold.to('sb_rate')
    
    # Checking user's choice for the mask_iter parameter
    if not isinstance(mask_iter, int):
        raise TypeError("The mask_iter argument must be an integer.")

    # Converting parameters from astropy units into a type the command line will accept
    pimin = int(pimin.value)
    pimax = int(pimax.value)
    mask_pimin = int(mask_pimin.value)
    mask_pimax = int(mask_pimax.value)
    timebin = float(timebin.value)
    source_size = float(source_size.value)

    # These parameters we want DAXA to have control over, not the user
    gridsize = 18   # Sections of the image a source detection is run over to determine a dynamic threshold
    fov_radius = 30 # Not sure about this parameter yet
    xmin = -108000  # These are for making the image
    xmax = 108000
    ymin = -108000
    ymax = 108000
    write_thresholdimg = 'yes'
    write_mask = 'yes'
    write_lightcurve = 'yes' 

    # Sets up storage dictionaries for bash commands, final file paths (to check they exist at the end), and any
    #  extra information that might be useful to provide to the next step in the generation process
    miss_cmds = {}
    miss_final_paths = {}
    miss_extras = {}

    # Just grabs the eROSITA missions, we already know there will be at least one because otherwise _esass_process_setup
    #  would have thrown an error
    erosita_miss = [mission for mission in obs_archive if mission.name in ALLOWED_EROSITA_MISSIONS]
    # We are iterating through XMM missions (options could include xmm_pointed and xmm_slew for instance).
    for miss in erosita_miss:
        # Sets up the top level keys (mission name) in our storage dictionaries
        miss_cmds[miss.name] = {}
        miss_final_paths[miss.name] = {}
        miss_extras[miss.name] = {}

        # This method will fetch the valid data (ObsID, Instruments) that can be processed
        all_obs_info = obs_archive.get_obs_to_process(miss.name)

        # DAVID_QUESTION cant see the step where obs with filterwheel closed are filtered out
        # Checking that any valid observations are left after the get_obs_to_process function is run
        if len(all_obs_info) == 0:
            raise FileNotFoundError("No valid observations have been found, so flaregti may not be run.")

        # We iterate through the valid identifying information
        for obs_info in all_obs_info:
            # JESS_TODO make sure the inst in this is a string of TMmodules used in ascending order 
            # Split out the information in obs_info
            obs_id, insts = obs_info

            # If all insts are used the name of the eventlist will be in a different format
            if len(insts) == 7:
                evt_list_file = miss._get_evlist_path_from_obs(obs=obs_id)
            else:
                evt_list_noinsts = miss._get_evlist_path_from_obs(obs=obs_id)
                # the :-5 removes the .fits at the end, so the correctly formatted version of the file name
                # with the instrument information can be appended
                evt_list_file = evt_list_noinsts[:-5] + '_if_{}.fits'.format(insts)

            # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.get_processed_data_path(miss, obs_id)

            # Set up a temporary directory to work in (probably not really necessary in this case, but will be
            #  in other processing functions).
            temp_name = "tempdir_{}".format(randint(0, 1e+8))
            temp_dir = dest_dir + temp_name + "/"

            # Setting up the paths to the gti file, lightcurve file, threshold image file, and mask image file
            og_gti_name = "{i}-gti.fits".format(i=insts)
            og_lc_name = "{i}-lc-{l}-{u}.fits".format(i=insts, l=pimin, u=pimax)
            og_thresholdimg_name = "{i}-thresholdimg-{l}-{u}.fits".format(i=insts, l=pimin, u=pimax)
            og_maskimg_name = "{i}-maskimg-{l}-{u}.fits".format(i=insts, l=mask_pimin, u=mask_pimax)
 
            gti_path = dest_dir + og_gti_name
            lc_path = dest_dir + og_lc_name
            threshold_path = dest_dir + og_thresholdimg_name
            maskimg_path = dest_dir + og_maskimg_name

            final_paths = [gti_path, lc_path, threshold_path, maskimg_path]

            # If it doesn't already exist then we will create commands to generate it
            # TODO Need to decide which file to check for here to see whether the command has already been run
            # Make the temporary directory (it shouldn't already exist but doing this to be safe)
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            flaregti_cmd = "cd {d}; flaregti eventfile={ef} gtifile={gtif} pimin={pimi} pimax={pima} " \
                 "mask_pimin{mpimi} mask_pimax={mpima} xmin={xmi} xmax={xma} ymin={ymi} ymax={yma} " \
                 "gridsize={gs} binsize={bs} detml={dl} timebin={tb} source_size={ss} source_like={sl} " \
                 "fov_radius={fr} threshold={t} max_threshold={mt} write_mask={wm} mask={m} mask_iter={mit} " \
                 "write_lightcurve={wl} lightcurve={lcf} write_thresholdimg={wti} thresholdimg={tif}" \
                 "; mv {ogti} ../{gti}; mv {olc} ../{lc}; mv {oti} ../{ti}; mv {omi} ../{mi}" \
                 "; rm -r {d}"

            cmd = flaregti_cmd.format(d=temp_dir, ef=evt_list_file, gtif=og_gti_name, pimi=pimin, pima=pimax,
                                      mpimi=mask_pimin, mpima=mask_pimax, xmi=xmin, xma=xmax, ymi=ymin,
                                      yma=ymax, gs=gridsize, bs=binsize, dl=detml, tb=timebin, ss=source_size,
                                      sl=source_like, fr=fov_radius, t=threshold, mt=max_threshold, wm=write_mask,
                                      m=og_maskimg_name, mit=mask_iter, wl=write_lightcurve, lcf=og_lc_name,
                                      wti=write_thresholdimg, tif=og_thresholdimg_name, ogti=og_gti_name,
                                      gti=gti_path, olc=og_lc_name, lc=lc_path, oti=og_thresholdimg_name,
                                      ti=threshold_path, omi=og_maskimg_name, mi=maskimg_path)

            # Now store the bash command, the path, and extra info in the dictionaries
            miss_cmds[miss.name][obs_id] = cmd
            miss_final_paths[miss.name][obs_id] = final_paths
            miss_extras[miss.name][obs_id] = {'gti_path': gti_path, 'lc_path': lc_path, 'threshold_path': threshold_path,
                                              'maskimg_path': maskimg_path}

    # This is just used for populating a progress bar during the process run
    process_message = 'Finding flares in eROSITA observations'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout