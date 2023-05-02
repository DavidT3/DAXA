# This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
# Last modified by David J Turner (turne540@msu.edu) Thu Apr 20 2023, 10:52. Copyright (c) The Contributors
import os
from random import randint

from astropy.units import Quantity

from daxa import NUM_CORES
from daxa.archive.base import Archive
from daxa.exceptions import NoDependencyProcessError
from daxa.process.erosita._common import _esass_process_setup, ALLOWED_EROSITA_MISSIONS

# JESS_TODO write an esass call wrapper and then the return and rtype in the docstring
# JESS_TODO check each argument type is correct

# DAVID_QUESTION need to discuss which arguments to keep

def flaregti(obs_archive: Archive, pimin: float = 200, pimax: float = 10000, mask_pimin: float = 200, 
            mask_pimax: float = 10000, xmin: float = -108000, xmax: float = 108000, ymin: float = -108000, 
            ymax: float = 108000, gridsize: int = 18, binsize: int = 1200, detml: int = 10, 
            timebin: int = 20, source_size: int = 25, source_like: int = 10, fov_radius: int = 30, 
            threshold: float = -1, max_threshold: float = -1, write_mask: bool = True, mask_iter: int = 3,
            write_lightcurve: bool = True, write_thresholdimg: bool = False, num_cores: int = NUM_CORES,
            disable_progress: bool = False, timeout: Quantity = None):
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
    :param float xmin: Sky pixel range for flare analysis: Xmin.
    :param float xmax: Sky pixel range for flare analysis: Xmax.
    :param float ymin: Sky pixel range for flare analysis: Ymin.
    :param float ymax: Sky pixel range for flare analysis: Ymax.
    :param int gridsize: Number of grid points per dimension for dynamic threshold calculation.
    :param int binsize: Bin size of mask image (unit: sky pixels).
    :param int detml: Likelihood threshold for mask creation.
    :param int timebin: Bin size for lightcurve (unit: seconds).
    :param int source_size: Diameter of source extracton area for dynamic threshold calculation (unit: arcsec);
        this is the most important parameter if optimizing for extended sources.
    :param int source_like: Source likelihood for automatic threshold calculation.
    :param int fov_radius: FoV radius used when computing a dynamic threshold (unit: arcmin).
    :param float threshold: Flare threshold; dynamic if negative (unit: counts/deg^2/sec).
    :param float max_threshold: Maximum threshold rate, if positive (unit: counts/deg^2/sec),
        if set this forces the threshold to be this rate or less.
    :param bool write_mask: Write mask image.
    :param int mask_iter: Number of repetitions of source masking and GTI creation.
    :param bool write_lightcurve: Write lightcurve.
    :param bool write_thresholdimg: Whether to write a FITS threshold image.
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
    
    # Checking that the upper energy limit is not below the lower energy limit (for the lightcurve)
    if pimax <= pimin:
        raise ValueError("The pimax argument must be larger than the pimin argument.")

    # Checking user's pimin and pimax inputs are in the valid energy range for eROSITA
    if (pimin < 200 or pimin > 10000) or (pimax < 200 or pimax > 10000):
        raise ValueError("The pimin and pimax value must be between 200 eV and 10000 eV.")
    
    # Checking that the upper energy limit is not below the lower energy limit (for the image)
    if mask_pimax <= mask_pimin:
        raise ValueError("The mask_pimax argument must be larger than the mask_pimin argument.")

    # Checking user's mask_pimin and mask_pimax inputs are in the valid energy range for eROSITA
    if (mask_pimin < 200 or mask_pimin > 10000) or (mask_pimax < 200 or mask_pimax > 10000):
        raise ValueError("The mask_pimin and mask_pimax value must be between 200 eV and 10000 eV.")

    # Checking xmin, xmax, ymin, ymax are valid for eSASS processing
    if (xmin < -108000 or xmin > 108000) or (xmax < -108000 or xmax > 108000) or \
       (ymin < -108000 or ymin > 108000) or (ymax < -108000 or ymax > 108000):
        raise ValueError("The xmin, xmax, ymin, and ymax values must be between -108000 and 108000.")
    
    # Checking that the higher pixel limit isn't below the lower limit in the x dimension
    if (xmax <= xmin):
        raise ValueError("The xmax argument must be larger than the xmin argument.")
    
    # Checking that the higher pixel limit isn't below the lower limit in the y dimension
    if (ymax <= ymin):
        raise ValueError("The ymax argument must be larger than the ymin argument.")
    
    # Checking user's input for write_thresholdimg is of the correct type
    if not isinstance(write_thresholdimg, bool):
        raise TypeError("The write_thresholdimg parameter must be a boolean.")

    # Checking user's input for write_mask is of the correct type
    if not isinstance(write_mask, bool):
        raise TypeError("The write_mask parameter must be a boolean.")
    
    # Checking user's input for write_lightcurve is of the correct type
    if not isinstance(write_lightcurve, bool):
        raise TypeError("The write_lightcurve parameter must be a boolean.")
    
    # Need to change parameter to write threshold image if the user wants it. The parameter
    #  must be changed from boolean to a 'yes' or 'no' string because that is what flaregti wants
    if write_thresholdimg:
        write_thresholdimg = 'yes'
    else:
        write_thresholdimg = 'no'
    # Similarly with write_mask
    if write_mask:
        write_mask = 'yes'
    else:
        write_mask = 'no'
    # And lastly with write_lightcurve
    if write_lightcurve:
        write_lightcurve = 'yes'
    else:
        write_lightcurve = 'no'

    # Checking string inputs are valid
    if write_thresholdimg != 'yes' or 'no':
        raise ValueError("The string passed for 'write_thresholdimg' must be either yes or no")
    if write_mask != 'yes' or 'no':
        raise ValueError("The string passed for 'write_mask' must be either yes or no")
    if write_lightcurve != 'yes' or 'no':
        raise ValueError("The string passed for 'write_lightcurve' must be either yes or no")

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
            # DAVID_QUESTION is this the right file?
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

            # DAVID_QUESTION why would you want your image energy limits and lightcurve limits to be different
            # Setting up the paths to the gti file, lightcurve file, threshold image file, and mask image file
            # DAVID_QUESTION not sure which energy lims to use?
            og_gti_name = "{i}-gti-{l}-{u}.fits".format(i=insts, l=pimin, u=pimax)
            og_lc_name = "{i}-lc-{l}-{u}.fits".format(i=insts, l=pimin, u=pimax)
            og_thresholdimg_name = "{i}-thresholdimg-{l}-{u}.fits".format(i=insts, l=pimin, u=pimax)
            og_maskimg_name = "{i}-thresholdimg-{l}-{u}.fits".format(i=insts, l=mask_pimin, u=mask_pimax)
 
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
            # DAVID_QUESTION not sure on what structure to store erosita data under
            miss_cmds[miss.name][obs_id] = cmd
            miss_final_paths[miss.name][obs_id] = final_paths
            miss_extras[miss.name][obs_id] = {'gti_path': gti_path, 'lc_path': lc_path, 'threshold_path': threshold_path,
                                              'maskimg_path': maskimg_path}

    # This is just used for populating a progress bar during the process run
    process_message = 'Finding flares in eROSITA observations'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout