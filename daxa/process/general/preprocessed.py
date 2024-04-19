#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 19/04/2024, 17:30. Copyright (c) The Contributors

from daxa.archive import Archive


def preprocessed_in_archive(arch: Archive):
    """
    This function acts on an archive's missions which were created with pre-processed data (with things like
    pre-generated event lists, images, and exposure maps downloaded when the archive was set up). It will take the
    existing products and re-organise/re-name them into DAXA's processed archive structure, with the DAXA file
    naming scheme.

    :param Archive arch: A DAXA archive that contains at least one mission with pre-processed data.
    """

    # This will iterate through all the missions associated with the passed archive which have pre-processed data, and
    #  if there are none a suitable error will be raised.
    for miss in arch.preprocessed_missions:
        print(miss)
        # pass

