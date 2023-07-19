# This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
# Last modified by Jessica Pilling (jp735@sussex.ac.uk) Wed Jul 19 2023, 13:52. Copyright (c) The Contributors

from astropy.units import Quantity

from daxa.archive.base import Archive
from daxa.process.erosita._common import _esass_process_setup, ALLOWED_EROSITA_MISSIONS, esass_call, ALL_EROSITA_FLAGS

@esass_call
def cleaned_evt_lists(obs_archive: Archive, lo_en: Quantity = None, hi_en: Quantity = None,
                      flag: int = 0xc0000000, flag_invert: bool = False, pattern: int = 15, num_cores: int = NUM_CORES,
                      disable_progress: bool = False, timeout: Quantity = None)

    """
    The function wraps the eROSITA eSASS task evtool, which is used for selecting events.
    This has been tested up to evtool v2.10.1

    This function is used to apply the soft-proton filtering (along with any other filtering you may desire, including
    the setting of energy limits) to eROSITA event lists, resulting in the creation of sets of cleaned event lists
    which are ready to be analysed (or merged together, if there are multiple exposures for a particular
    observation-instrument combination).

    :param Archive obs_archive: An Archive instance containing eROSITA mission instances with observations for
        which cleaned event lists should be created. This function will fail if no eROSITA missions are present in the archive.
    :param Quantity lo_en: The lower bound of an energy filter to be applied to the cleaned, filtered, event lists. If
        'lo_en' is set to an Astropy Quantity, then 'hi_en' must be as well. Default is None, in which case no
        energy filter is applied.
    :param Quantity hi_en: The upper bound of an energy filter to be applied to the cleaned, filtered, event lists. If
        'hi_en' is set to an Astropy Quantity, then 'lo_en' must be as well. Default is None, in which case no
        energy filter is applied.
    :param int flag: FLAG parameter to select events based on owner, information, rejection, quality, and corrupted data. The eROSITA
        website contains the full description of event flags. The default parameter will remove all events flagged as either singly 
        corrupt or as part of a corrupt frame.
    :param bool flag_invert: If set to True, this function will discard all events selected by the flag parameter.
    :param int pattern: Selects events of a certain pattern chosen by the integer key. The default of 15 selects all four of the
        recognized legal patterns.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire cleaned_evt_lists process, but a timeout for individual
        ObsID-Inst-subexposure processes.
    """

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

    # Checking user has input the flag parameter as a string
    if not isinstance(flag, int):
            raise TypeError("The flag parameter must be an integer.")

    # Checking the input is a valid hexidecimal number
    def is_hexadecimal(s):
        try:
            # if the number is hexidecimal then it will be an integer with base 16
            int(str(s), 16)
            return True
        except ValueError:
            return False
    
    if not is_hexadecimal(flag):
            raise ValueError("The flag parameter must be a valid hexidecimal number.")
    
    # Checking user has input a valid erosita flag
    if flag > 2**32:
        raise ValueError("The flag parameter is not a valid eROSITA flag.")
    
    # Checking user has input flag_invert as a boolean
    if not isinstance(flag_invert, bool):
        raise TypeError("The flag_invert parameter must be a boolean.")
    
    # Checking user has input pattern as an integer
    if not isinstance(pattern, int):
        raise TypeError("The pattern parameter must be a boolean between 1 and 15 inclusive.")
    
    #  Checking user has input a valid pattern
    if (pattern <= 0 or pattern >= 16):
        raise ValueError("Valid eROSITA patterns are between 1 and 15 inclusive")

    

    

        