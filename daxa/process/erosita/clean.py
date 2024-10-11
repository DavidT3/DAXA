#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 11/10/2024, 17:07. Copyright (c) The Contributors
import os
from random import randint
from typing import Union

from astropy.units import Quantity, UnitConversionError, add_enabled_units

from daxa import NUM_CORES, sb_rate
from daxa.archive.base import Archive
from daxa.exceptions import NoProcessingError
from daxa.process.erosita._common import _esass_process_setup, ALLOWED_EROSITA_MISSIONS, esass_call

# Adding this to the enabled astropy units so that it can be used in flaregti to define thresholds
add_enabled_units([sb_rate])


@esass_call
def flaregti(obs_archive: Archive, pimin: Quantity = Quantity(200, 'eV'), pimax: Quantity = Quantity(10000, 'eV'),
             mask_pimin: Quantity = Quantity(200, 'eV'), mask_pimax: Quantity = Quantity(10000, 'eV'),
             binsize: int = 1200, detml: Union[float, int] = 10, timebin: Quantity = Quantity(20, 's'),
             source_size: Quantity = Quantity(25, 'arcsec'), source_like: Union[float, int] = 10,
             threshold: Quantity = Quantity(-1, 'ct/(deg^2 * s)'),
             max_threshold: Quantity = Quantity(-1, 'ct/(deg^2 * s)'),
             mask_iter: int = 3, num_cores: int = NUM_CORES, disable_progress: bool = False, timeout: Quantity = None):
    """
    The DAXA wrapper for the eROSITA eSASS task flaregti, which attempts to identify good time intervals with
    minimal flaring. This has been tested up to flaregti v1.20.

    This function does not generate final event lists, but instead is used to create good-time-interval files
    which are then applied to the creation of final event lists, along with other user-specified filters, in the
    'cleaned_evt_lists' function.

    :param Archive obs_archive: An Archive instance containing eROSITA mission instances with observations for
        which flaregti should be run. This function will fail if no eROSITA missions are present in the archive.
    :param float pimin:  Lower PI bound of energy range for lightcurve creation.
    :param float pimax:  Upper PI bound of energy range for lightcurve creation.
    :param float mask_pimin: Lower PI bound of energy range for finding sources to mask.
    :param float mask_pimax: Upper PI bound of energy range for finding sources to mask.
    :param int binsize: Bin size of mask image (unit: sky pixels).
    :param int detml: Likelihood threshold for mask creation.
    :param int timebin: Bin size for lightcurve (unit: seconds).
    :param int source_size: Diameter of source extraction area for dynamic threshold calculation (unit: arcsec);
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
    if (mask_pimin < Quantity(200, 'eV') or mask_pimin > Quantity(10000, 'eV')) or \
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
    if timebin > Quantity(1000000000, 's'):
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
    elif not threshold.unit.is_equivalent('sb_rate'):
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
    elif not max_threshold.unit.is_equivalent('sb_rate'):
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
    threshold = threshold.value
    max_threshold = max_threshold.value

    # These parameters we want DAXA to have control over, not the user
    gridsize = 18   # Sections of the image a source detection is run over to determine a dynamic threshold
    fov_radius = 30  # Not sure about this parameter yet
    xmin = -108000  # These are for making the image
    xmax = 108000
    ymin = -108000
    ymax = 108000
    write_thresholdimg = 'yes'
    write_mask = 'yes'
    write_lightcurve = 'yes' 

    # Defining the command - we create a symlink to the event list primarily because we had some issues with
    #  flaregti being able to read in the event lists, but only on some systems - this seems more consistent
    flaregti_cmd = "cd {d}; ln -s {ef} {lef}; flaregti eventfile={lef} pimin={pimi} pimax={pima} " \
                   "mask_pimin={mpimi} mask_pimax={mpima} xmin={xmi} xmax={xma} ymin={ymi} ymax={yma} " \
                   "gridsize={gs} binsize={bs} detml={dl} timebin={tb} source_size={ss} source_like={sl} " \
                   "fov_radius={fr} threshold={t} max_threshold={mt} write_mask={wm} mask={m} mask_iter={mit} " \
                   "write_lightcurve={wl} lightcurve={lcf} write_thresholdimg={wti} thresholdimg={tif}; " \
                   "mv {olc} {lc}; mv {oti} {ti}; mv {omi} {mi}; rm -r {d}"

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
            
        # Checking that any valid observations are left after the get_obs_to_process function is run
        if len(all_obs_info) == 0:
            raise FileNotFoundError("No valid observations have been found, so flaregti may not be run.")

        # all_obs_info is a list of lists, where each list is of the format: [ObsID, Inst, 'usable'].
        # There is a new list for each instrument, but I just want to loop over the ObsID in the following bit of code,
        #  I also want to know all the instruments that the ObsID contains events for
        #  So here I am just making a dictionary of the format: {ObsID: insts}
        # Getting unique obs_ids in all_obs_info
        obs_ids = list(set([all_obs_info_list[0] for all_obs_info_list in all_obs_info]))
        obs_info_dict = {}
        for obs in obs_ids:
            # Collecting all the insts that a certain ObsID has events for
            insts = ''.join([all_obs_info_list[1] for all_obs_info_list in all_obs_info if obs in all_obs_info_list])

            # The insts are all TM{x} where x is a number from 1-7, we want to separate them with _ for the file names
            obs_info_dict[obs] = '_'.join("TM"+ch for ch in insts if ch.isdigit())

        # We iterate through the valid identifying information
        for obs_id in obs_info_dict:

            # Getting the insts associated with this obs for file naming purposes 
            insts = obs_info_dict[obs_id]

            # Search through the process_extra_info attribute of the archive to find the paths 
            #   to the event lists
            evt_list_file = obs_archive[miss.name].get_evt_list_path(obs_id)

            # This path is guaranteed to exist, as it was set up in _esass_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.construct_processed_data_path(miss, obs_id)
            # Set up a temporary directory to work in (probably not really necessary in this case, but will be
            #  in other processing functions).
            temp_name = "tempdir_{}".format(randint(0, int(1e+8)))
            temp_dir = dest_dir + temp_name + "/"

            # Setting up the paths to the lightcurve file, threshold image file, and mask image file
            og_lc_name = "{i}-lc-{l}-{u}.fits".format(i=insts, l=pimin, u=pimax)
            og_thresholdimg_name = "{i}-thresholdimg-{l}-{u}.fits".format(i=insts, l=pimin, u=pimax)
            og_maskimg_name = "{i}-maskimg-{l}-{u}.fits".format(i=insts, l=mask_pimin, u=mask_pimax)
 
            lc_name = "obsid{oi}-inst{i}-subexpNone-en{l}_{h}PI-lightcurve.fits".format(i=insts, l=pimin, h=pimax,
                                                                                        oi=obs_id)
            thresholdimg_name = "obsid{oi}-inst{i}-subexpNone-en{l}_{h}PI-thresholdimage.fits".format(i=insts, l=pimin,
                                                                                                      h=pimax,
                                                                                                      oi=obs_id)
            maskimg_name = "obsid{oi}-inst{i}-subexpNone-en{l}_{h}PI-maskimage.fits".format(i=insts, l=mask_pimin,
                                                                                            h=mask_pimax, oi=obs_id)

            lc_path = os.path.join(dest_dir, 'cleaning', lc_name)
            threshold_path = os.path.join(dest_dir, 'cleaning', thresholdimg_name)
            maskimg_path = os.path.join(dest_dir, 'cleaning', maskimg_name)

            final_paths = [lc_path, threshold_path, maskimg_path]

            # As this is the first process in the chain, we need to account for the fact that nothing has been run
            #  before, and using the process_success property might raise an exception
            try:
                check_dict = obs_archive.process_success[miss.name]['flaregti']
            except (NoProcessingError, KeyError):
                check_dict = {}

            # If it doesn't already exist then we will create commands to generate it
            if obs_id not in check_dict:
                # Make the temporary directory (it shouldn't already exist but doing this to be safe)
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                cmd = flaregti_cmd.format(d=temp_dir, lef="temp_{oi}_evt_pth".format(oi=obs_id), ef=evt_list_file,
                                          pimi=pimin, pima=pimax, mpimi=mask_pimin, mpima=mask_pimax, xmi=xmin,
                                          xma=xmax, ymi=ymin, yma=ymax, gs=gridsize, bs=binsize, dl=detml, tb=timebin,
                                          ss=source_size, sl=source_like, fr=fov_radius, t=threshold, mt=max_threshold,
                                          wm=write_mask, m=og_maskimg_name, mit=mask_iter, wl=write_lightcurve,
                                          lcf=og_lc_name, wti=write_thresholdimg, tif=og_thresholdimg_name,
                                          olc=og_lc_name, lc=lc_path, oti=og_thresholdimg_name, ti=threshold_path,
                                          omi=og_maskimg_name, mi=maskimg_path)

                # Now store the bash command, the path, and extra info in the dictionaries
                miss_cmds[miss.name][obs_id] = cmd
                miss_final_paths[miss.name][obs_id] = final_paths
                miss_extras[miss.name][obs_id] = {'lc_path': lc_path, 'threshold_path': threshold_path,
                                                  'maskimg_path': maskimg_path}

    # This is just used for populating a progress bar during the process run
    process_message = 'Finding flares in observations'

    return (miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress,
            timeout, esass_in_docker)