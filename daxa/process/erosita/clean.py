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
            threshold: float = -1, max_threshold: float = -1, write_mask: str = 'yes', mask_iter: int = 3,
            write_lightcurve: str = 'yes', write_thresholdimg: str = 'no'):
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
    :param write_mask str: Write mask image.
    :param mask_iter int: Number of repetitions of source masking and GTI creation.
    :param write_lightcurve str: Write lightcurve.
    :param write_thresholdimg str: Whether to write a FITS threshold image.
    """
    pass