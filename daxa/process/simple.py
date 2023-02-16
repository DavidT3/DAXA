#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 27/01/2023, 16:49. Copyright (c) The Contributors

from astropy.units import Quantity

from daxa import NUM_CORES
from daxa.archive.base import Archive
from daxa.process.xmm.assemble import epchain, emchain, cleaned_evt_lists, merge_subexposures
from daxa.process.xmm.check import emanom
from daxa.process.xmm.clean import espfilt
from daxa.process.xmm.setup import cif_build, odf_ingest


def full_process_xmm(obs_archive: Archive, lo_en: Quantity = None, hi_en: Quantity = None,
                     process_unscheduled: bool = True, num_cores: int = NUM_CORES, timeout: Quantity = None):
    """
    This is a convenience function that will fully process and prepare XMM data in an archive using the default
    configuration settings of all the cleaning steps. If you wish to exercise finer grained control over the
    processing of your data then you can copy the steps of this function and alter the various parameter values.

    :param Archive obs_archive: An archive object that contains at least one XMM mission to be processed.
    :param Quantity lo_en: If an energy filter should be applied to the final cleaned event lists, this is the
        lower energy bound. The default is None, in which case NO ENERGY FILTER is applied.
    :param Quantity hi_en: If an energy filter should be applied to the final cleaned event lists, this is the
        upper energy bound. The default is None, in which case NO ENERGY FILTER is applied.
    :param bool process_unscheduled: Should unscheduled sub-exposures be processed and included in the final event
        lists. The default is True.
    :param int num_cores: The number of cores that can be used by the processing functions. The default is set to
        the DAXA NUM_CORES parameter, which is configured to be 90% of the system's cores.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire processing stack, but a timeout for the individual
        processes of each stage, whether they are at the ObsID, ObsID-Inst, or ObsID-Inst-Subexposure level of
        granularity.
    """
    # Creates calibration files for the XMM observations
    cif_build(obs_archive, num_cores=num_cores, timeout=timeout)
    # Prepares the summary files for the XMM observations - used by processes to determine what data there are
    #  for each observation
    odf_ingest(obs_archive, num_cores=num_cores, timeout=timeout)
    # This step combines the separate CCD data for any EPIC-PN observations
    epchain(obs_archive, process_unscheduled, num_cores=num_cores, timeout=timeout)
    # This step does much the same but for EPIC-MOS observations
    emchain(obs_archive, process_unscheduled, num_cores=num_cores, timeout=timeout)
    # This checks for anomalous CCD states in MOS observations - this step isn't obligatory but is
    #  probably a good idea
    emanom(obs_archive, num_cores=num_cores, timeout=timeout)
    # Runs through all available data and checks for periods of soft-proton flaring - this information is used by the
    #  cleaned event lists function to remove those time periods as part of the cleaning/filtering process
    espfilt(obs_archive, num_cores=num_cores, timeout=timeout)
    # Creates the cleaned event lists, with SP flaring removed and standard event filters (filtering on pattern etc.)
    #  applied to both EPIC-PN and EPIC-MOS data.
    cleaned_evt_lists(obs_archive, lo_en, hi_en, num_cores=num_cores, timeout=timeout)
    # Finally this function checks for cases where an ObsID-instrument combination has sub-exposures that should be
    #  merged into a single, final, event list.
    merge_subexposures(obs_archive, num_cores=num_cores, timeout=timeout)
