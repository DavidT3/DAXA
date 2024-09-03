#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 02/09/2024, 21:09. Copyright (c) The Contributors

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
                            'odf_ingest': odf_ingest}
               }