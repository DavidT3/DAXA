#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 15/11/2024, 00:11. Copyright (c) The Contributors

from astropy.units import Quantity

from daxa import NUM_CORES
from daxa.archive.base import Archive
from daxa.process import chandra_repro, cleaned_chandra_evts, flux_image, rate_image, nupipeline_calibrate
from daxa.process.chandra import prepare_chandra_info
from daxa.process.chandra.clean import deflare
from daxa.process.erosita.assemble import cleaned_evt_lists as eros_cleaned_evt_lists
from daxa.process.erosita.clean import flaregti
from daxa.process.nustar.setup import prepare_nustar_info
from daxa.process.xmm._common import ALLOWED_XMM_MISSIONS
from daxa.process.xmm.assemble import (epchain, emchain, cleaned_evt_lists, merge_subexposures, rgs_events,
                                       rgs_angles, cleaned_rgs_event_lists)
from daxa.process.xmm.check import emanom
from daxa.process.xmm.clean import espfilt
from daxa.process.xmm.setup import cif_build, odf_ingest


def full_process_xmm(obs_archive: Archive, lo_en: Quantity = None, hi_en: Quantity = None,
                     process_unscheduled: bool = True, find_mos_anom_state: bool = False,
                     num_cores: int = NUM_CORES, timeout: Quantity = None):
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
    :param bool find_mos_anom_state: Whether the emanom task should be run to search for anomolous states of the MOS
        cameras. This is set to False by default, and you should be aware that I have not found it to be particularly
        reliable with the default settings, it tends to remove good chips.
    :param int num_cores: The number of cores that can be used by the processing functions. The default is set to
        the DAXA NUM_CORES parameter, which is configured to be 90% of the system's cores.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire processing stack, but a timeout for the individual
        processes of each stage, whether they are at the ObsID, ObsID-Inst, or ObsID-Inst-Subexposure level of
        granularity.
    """
    from daxa.process.xmm.generate import generate_images_expmaps

    # Creates calibration files for the XMM observations
    cif_build(obs_archive, num_cores=num_cores, timeout=timeout)
    # Prepares the summary files for the XMM observations - used by processes to determine what data there are
    #  for each observation
    odf_ingest(obs_archive, num_cores=num_cores, timeout=timeout)

    # It is very much not a given that there will be RGS data to process, so we first check to see if any of the
    #  XMM missions have actually had RGS data selected
    with_rgs = any(['R1' in mission.chosen_instruments or 'R2' in mission.chosen_instruments for mission in obs_archive
                    if mission.name in ALLOWED_XMM_MISSIONS])

    # If RGS has been selected then we will try to process it, but in case there are no actual data we do put a
    #  try-except
    if with_rgs:
        try:
            rgs_events(obs_archive, process_unscheduled, num_cores=num_cores, timeout=timeout)
            rgs_angles(obs_archive, num_cores=num_cores, timeout=timeout)
            cleaned_rgs_event_lists(obs_archive, num_cores=num_cores, timeout=timeout)
        except (ValueError, IndexError):
            pass

    # We try to process EPIC PN data, but we use a try-except because it is possible that none will have been
    #  selected when the mission was defined
    try:
        # This step combines the separate CCD data for any EPIC-PN observations
        epchain(obs_archive, process_unscheduled, num_cores=num_cores, timeout=timeout)
    except ValueError:
        pass
    # Same deal but for the MOS data - all the succeeding steps should handle this by themselves, checking for
    #  the data that are present
    try:
        # This step does much the same but for EPIC-MOS observations
        emchain(obs_archive, process_unscheduled, num_cores=num_cores, timeout=timeout)
        em_data = True
    except ValueError:
        em_data = False

    # The user can choose whether this state is run, if it isn't then cleaned_evt_lists should automatically
    #  turn off its filtering based on anomolous state codes. Also if there are actually MOS data to use
    if find_mos_anom_state and em_data:
        # This checks for anomalous CCD states in MOS observations - this step isn't obligatory but is
        #  probably a good idea
        emanom(obs_archive, num_cores=num_cores, timeout=timeout)

    # Runs through all available data and checks for periods of soft-proton flaring - this information is used by the
    #  cleaned event lists function to remove those time periods as part of the cleaning/filtering process
    espfilt(obs_archive, num_cores=num_cores, timeout=timeout)
    # Creates the cleaned event lists, with SP flaring removed and standard event filters (filtering on pattern etc.)
    #  applied to both EPIC-PN and EPIC-MOS data.
    cleaned_evt_lists(obs_archive, lo_en, hi_en, filt_mos_anom_state=find_mos_anom_state, num_cores=num_cores,
                      timeout=timeout)
    # Finally this function checks for cases where an ObsID-instrument combination has sub-exposures that should be
    #  merged into a single, final, event list.
    merge_subexposures(obs_archive, num_cores=num_cores, timeout=timeout)

    # Also added the automatic generation of 0.5-2.0 and 2.0-10.0 keV images and exposure maps
    generate_images_expmaps(obs_archive, num_cores=num_cores)


def full_process_erosita(obs_archive: Archive, lo_en: Quantity = None, hi_en: Quantity = None,
                         num_cores: int = NUM_CORES, timeout: Quantity = None):
    """
    This is a convenience function that will fully process and prepare eROSITA data in an archive using the default
    configuration settings of all the cleaning steps. If you wish to exercise finer grained control over the
    processing of your data then you can copy the steps of this function and alter the various parameter values.

    :param Archive obs_archive: An archive object that contains at least one eROSITA mission to be processed.
    :param Quantity lo_en: If an energy filter should be applied to the final cleaned event lists, this is the
        lower energy bound. The default is None, in which case NO ENERGY FILTER is applied.
    :param Quantity hi_en: If an energy filter should be applied to the final cleaned event lists, this is the
        upper energy bound. The default is None, in which case NO ENERGY FILTER is applied.
    :param int num_cores: The number of cores that can be used by the processing functions. The default is set to
        the DAXA NUM_CORES parameter, which is configured to be 90% of the system's cores.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire processing stack, but a timeout for the individual
        processes of each stage.
    """

    # This tool attempts to automatically remove any time periods that are heavily affected by soft-proton flaring
    flaregti(obs_archive, num_cores=num_cores, timeout=timeout)
    # Creates final cleaned event lists for eROSITA missions
    eros_cleaned_evt_lists(obs_archive, lo_en, hi_en, num_cores=num_cores, timeout=timeout)

    # Also added the automatic generation of 0.5-2.0 and 2.0-10.0 keV images and exposure maps
    # generate_images_expmaps(obs_archive, num_cores=num_cores)


def full_process_chandra(obs_archive: Archive, evt_lo_en: Quantity = None, evt_hi_en: Quantity = None,
                         num_cores: int = NUM_CORES, timeout: Quantity = None):
    """
    This is a convenience function that will fully process and prepare Chandra data in an archive using the default
    configuration settings of all the cleaning steps. If you wish to exercise finer grained control over the
    processing of your data then you can copy the steps of this function and alter the various parameter values.

    :param Archive obs_archive: An archive object that contains at least one Chandra mission to be processed.
    :param Quantity evt_lo_en: If an energy filter should be applied to the final cleaned event lists, this is the
        lower energy bound. The default is None, in which case NO ENERGY FILTER is applied.
    :param Quantity evt_hi_en: If an energy filter should be applied to the final cleaned event lists, this is the
        upper energy bound. The default is None, in which case NO ENERGY FILTER is applied.
    :param int num_cores: The number of cores that can be used by the processing functions. The default is set to
        the DAXA NUM_CORES parameter, which is configured to be 90% of the system's cores.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire processing stack, but a timeout for the individual
        processes of each stage, whether they are at the ObsID or ObsID-Inst level of granularity.
    """

    # Firstly we run this function, which parses all the observation index files (OIFs) for the Chandra data
    #  in our archive - prepares the observation_summaries which are then used by other processing steps
    prepare_chandra_info(obs_archive)

    # This is the next step, essentially a DAXA version/wrapping of the incredibly handy (thank you CXC)
    #  Chandra reprocessing script (chandra_repro). This will automatically generate level-2 event lists, as well
    #  as other necessary products, applying the current calibration.
    # I won't let the user pass anything to this, as the arguments control relatively minor things, and they can
    #  run it themselves if they want to.
    chandra_repro(obs_archive, num_cores=num_cores, timeout=timeout)

    # Now we use the 'deflare' function to attempt to automatically find any flaring in the Chandra observations, and
    #  to create GTIs that will allow us to remove it
    deflare(obs_archive, num_cores=num_cores, timeout=timeout)

    # Now the flaring GTIs are used to create final cleaned event lists - we do allow the user to control the
    #  cleaned event list energy bands, a departure from the default behaviour
    cleaned_chandra_evts(obs_archive, lo_en=evt_lo_en, hi_en=evt_hi_en, num_cores=num_cores, timeout=timeout)

    # We run the flux image function, and the rate image function, to create a TONNE of photometric products. This
    #  makes sure we have normal images, flux and rate maps, weighted and standard exposure maps, and PSF radius
    #  maps - all in the default Chandra CSC bands
    flux_image(obs_archive, num_cores=num_cores, timeout=timeout)
    rate_image(obs_archive, num_cores=num_cores, timeout=timeout)


def full_process_nustar(obs_archive: Archive, evt_lo_en: Quantity = None, evt_hi_en: Quantity = None,
                        num_cores: int = NUM_CORES, timeout: Quantity = None):
    """
    This is a convenience function that will fully process and prepare NuSTAR data in an archive using the default
    configuration settings of all the cleaning steps. If you wish to exercise finer grained control over the
    processing of your data then you can copy the steps of this function and alter the various parameter values.

    :param Archive obs_archive: An archive object that contains at least one NuSTAR mission to be processed.
    :param Quantity evt_lo_en: If an energy filter should be applied to the final cleaned event lists, this is the
        lower energy bound. The default is None, in which case NO ENERGY FILTER is applied.
    :param Quantity evt_hi_en: If an energy filter should be applied to the final cleaned event lists, this is the
        upper energy bound. The default is None, in which case NO ENERGY FILTER is applied.
    :param int num_cores: The number of cores that can be used by the processing functions. The default is set to
        the DAXA NUM_CORES parameter, which is configured to be 90% of the system's cores.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire processing stack, but a timeout for the individual
        processes of each stage, whether they are at the ObsID or ObsID-Inst level of granularity.
    """

    # Firstly we run this function, which parses the observation catalog files that ship with NuSTAR data to
    #  ascertain if the data can be used - frankly this doesn't do that much in NuSTAR's case
    prepare_nustar_info(obs_archive)

    # Now we run the first stage of the NuSTARDAS nupipeline tool - this prepares the base data and necessary
    #  files - applying the latest calibrations to the event lists
    nupipeline_calibrate(obs_archive, num_cores=num_cores, timeout=timeout)

