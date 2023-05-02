# This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
# Last modified by David J Turner (turne540@msu.edu) Thu Apr 20 2023, 10:52. Copyright (c) The Contributors

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
            write_lightcurve: bool = True, write_thresholdimg: bool = False):
    """
    The DAXA wrapper for the eROSITA eSASS task flaregti, which attempts to identify good time intervals with minimal flaring.
    This has been tested up to flaregti v1.20.

    This function does not generate final event lists, but instead is used to create good-time-interval files
    which are then applied to the creation of final event lists, along with other user-specified filters, in the
    'cleaned_evt_lists' function.

    :param Archive obs_archive: An Archive instance containing eROSITA mission instances with observations for
        which flaregti should be run. This function will fail if no eROSITA missions are present in the archive.
    :param pimin float:  Lower PI bound of energy range for lightcurve creation.
    :param pimax float:  Upper PI bound of energy range for lightcurve creation.
    :param mask_pimin: Lower PI bound of energy range for finding sources to mask.
    :param mask_pimax: Upper PI bound of energy range for finding sources to mask.
    :param xmin float: Sky pixel range for flare analysis: Xmin.
    :param xmax float: Sky pixel range for flare analysis: Xmax.
    :param ymin float: Sky pixel range for flare analysis: Ymin.
    :param ymax float: Sky pixel range for flare analysis: Ymax.
    :param gridsize int: Number of grid points per dimension for dynamic threshold calculation.
    :param binsize int: Bin size of mask image (unit: sky pixels).
    :param detml int: Likelihood threshold for mask creation.
    :param timebin int: Bin size for lightcurve (unit: seconds).
    :param source_size int: Diameter of source extracton area for dynamic threshold calculation (unit: arcsec);
        this is the most important parameter if optimizing for extended sources.
    :param source_like int: Source likelihood for automatic threshold calculation.
    :param fov_radius int: FoV radius used when computing a dynamic threshold (unit: arcmin).
    :param threshold float: Flare threshold; dynamic if negative (unit: counts/deg^2/sec).
    :param max_threshold float: Maximum threshold rate, if positive (unit: counts/deg^2/sec),
        if set this forces the threshold to be this rate or less.
    :param write_mask bool: Write mask image.
    :param mask_iter int: Number of repetitions of source masking and GTI creation.
    :param write_lightcurve bool: Write lightcurve.
    :param write_thresholdimg bool: Whether to write a FITS threshold image.
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



    pass