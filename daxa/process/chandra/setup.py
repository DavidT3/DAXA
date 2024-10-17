#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 17/10/2024, 16:25. Copyright (c) The Contributors


def parse_oif(sum_path: str, obs_id: str = None):
    """
    A function that takes a path to a Chandra 'oif.fits' file included in each ObsID directory on the remote
    archive. The file will be filtered and parsed so that data relevant to DAXA processing valid scientific
    observations can be extracted. This includes things like which mode the detector was in, whether a grating
    was deployed, etc.

    :param str sum_path: The path to the Chandra 'oif.fits' file that is to be parsed into a dictionary
        of relevant information.
    :param str obs_id: Optionally, the observation ID that goes with this summary file can be passed, purely to
        make a possible error message more useful.
    :return: Multi-level dictionary of information.
    :rtype: dict
    """
    # TODO ADD MORE INFORMATION TO THE 'return' PARAM IN THE DOCSTRING
    pass
