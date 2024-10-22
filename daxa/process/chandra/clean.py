#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 21/10/2024, 23:33. Copyright (c) The Contributors

from typing import Tuple

from astropy.units import Quantity

from daxa import NUM_CORES
from daxa.archive import Archive
from daxa.process.chandra._common import ciao_call, _ciao_process_setup


@ciao_call
def deflare(obs_archive: Archive, method: str = 'sigma', allowed_sigma: float = 3.0, min_length: int = 3,
            num_cores: int = NUM_CORES, disable_progress: bool = False, timeout: Quantity = None):
    """

    method: str = 'histogram', with_smoothing: Union[bool, Quantity] = True,
            with_binning: Union[bool, Quantity] = True, ratio: float = 1.2,
            filter_lo_en: Quantity = Quantity(2500, 'eV'), filter_hi_en: Quantity = Quantity(8500, 'eV'),
            range_scale: dict = None, allowed_sigma: float = 3.0, gauss_fit_lims: Tuple[float, float] = (0.1, 6.5),
            num_cores: int = NUM_CORES, disable_progress: bool = False, timeout: Quantity = None

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
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire espfilt process, but a timeout for individual
        ObsID-Inst-subexposure processes.
    :return: Information required by the SAS decorator that will run commands. Top level keys of any dictionaries are
        internal DAXA mission names, next level keys are ObsIDs. The return is a tuple containing a) a dictionary of
        bash commands, b) a dictionary of final output paths to check, c) a dictionary of extra info (in this case
        obs and analysis dates), d) a generation message for the progress bar, e) the number of cores allowed, and
        f) whether the progress bar should be hidden or not.
    :rtype: Tuple[dict, dict, dict, str, int, bool, Quantity]
    """
    # Run the setup for Chandra processes, which checks that CIAO is installed (as well as CALDB), and checks that the
    #  archive has at least one Chandra mission in it, and
    ciao_vers, caldb_vers, chan_miss = _ciao_process_setup(obs_archive)

    # We're going to wrap the 'deflare' tool that is included in CIAO, which will make our shiny new GTIs that
    #  exclude periods of intense soft-proton flaring
    crp_cmd = ("cd {d}; deflare infile={in_f} outfile={out_f} method={me} minlength={ml} verbose=5")

    # mv {oge} {fe}; mv {oggti} {fgti}; mv {ogbp} {fbp}; mv {ogfov} {ffov};
    # cd ..; rm -r {d}

    # ---------------------------------- Checking and converting user inputs ----------------------------------
    # Make sure that the method is one of the two allowed options, defined by the two types of cleaning
    #  that the deflare tool can do
    if method not in ['sigma', 'clean']:
        raise ValueError("The 'method' must be either 'sigma', or 'clean'.")
    elif method == 'clean':
        raise NotImplementedError("The lightcurve cleaning method 'lc_clean' is not yet fully supported by "
                                  "DAXA, please contact the developers if you need this functionality.")

    # ---------------------------------------------------------------------------------------------------------

