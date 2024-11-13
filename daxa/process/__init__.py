#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 12/11/2024, 22:17. Copyright (c) The Contributors

from daxa.process.chandra import prepare_chandra_info
from daxa.process.chandra.assemble import chandra_repro, cleaned_chandra_evts
from daxa.process.chandra.clean import deflare
from daxa.process.chandra.generate import flux_image, rate_image
from daxa.process.erosita import flaregti
from daxa.process.erosita.assemble import cleaned_evt_lists as ecleaned_evt_lists
from daxa.process.nustar.assemble import nupipeline_calibrate
from daxa.process.nustar.setup import prepare_nustar_info
from daxa.process.xmm import epchain, emchain, rgs_events, rgs_angles, cleaned_rgs_event_lists, cleaned_evt_lists, \
    merge_subexposures, emanom, espfilt, cif_build, odf_ingest

# TODO NEED TO ADD TO THE CHANDRA ENTRY AS I GO ALONG WITH THIS
PROC_LOOKUP = {'xmm_pointed': {'epchain': epchain,
                               'emchain': emchain,
                               'rgs_events': rgs_events,
                               'rgs_angles': rgs_angles,
                               'cleaned_rgs_event_lists': cleaned_rgs_event_lists,
                               'cleaned_evt_lists': cleaned_evt_lists,
                               'merge_subexposures': merge_subexposures,
                               'emanom': emanom,
                               'espfilt': espfilt,
                               'cif_build': cif_build,
                               'odf_ingest': odf_ingest},

               'xmm_slew': {'epchain': epchain,
                            'emchain': emchain,
                            'rgs_events': rgs_events,
                            'rgs_angles': rgs_angles,
                            'cleaned_rgs_event_lists': cleaned_rgs_event_lists,
                            'cleaned_evt_lists': cleaned_evt_lists,
                            'merge_subexposures': merge_subexposures,
                            'emanom': emanom,
                            'espfilt': espfilt,
                            'cif_build': cif_build,
                            'odf_ingest': odf_ingest},

               'erosita_calpv': {'cleaned_evt_lists': ecleaned_evt_lists,
                                 'flaregti': flaregti},

               'erosita_all_sky_de_dr1': {'cleaned_evt_lists': ecleaned_evt_lists,
                                          'flaregti': flaregti},

               'nustar_pointed': {'prepare_nustar_info': prepare_nustar_info,
                                  'nupipeline_calibrate': nupipeline_calibrate},

               'chandra': {'prepare_chandra_info': prepare_chandra_info,
                           'chandra_repro': chandra_repro,
                           'deflare': deflare,
                           'cleaned_chandra_evts': cleaned_chandra_evts,
                           'flux_image': flux_image,
                           'rate_image': rate_image},

               'rosat_all_sky': {},
               'rosat_pointed': {},
               'swift': {},
               'suzaku': {},
               'asca': {},
               'integral_pointed': {}
               }
