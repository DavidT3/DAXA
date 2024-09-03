#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 03/09/2024, 10:20. Copyright (c) The Contributors

from daxa.process.xmm import epchain, emchain, rgs_events, rgs_angles, cleaned_rgs_event_lists, cleaned_evt_lists, \
    merge_subexposures, emanom, espfilt, cif_build, odf_ingest

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
               'nustar_pointed': {},
               'chandra': {},
               'erosita_calpv': {},
               'erosita_all_sky_de_dr1': {},
               'rosat_all_sky': {},
               'rosat_pointed': {},
               'swift': {},
               'suzaku': {},
               'asca': {},
               'integral_pointed': {}
               }
