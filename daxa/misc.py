#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 08/12/2022, 10:17. Copyright (c) The Contributors

# Defining XGA information here, the write_xga_config method of the Archive class is written so that
# versions of XGA are released with more telescopes, developers only have to add the new config file
# below, and append this to the XGA_VER_DICT

from packaging.version import Version

# Config file for xga < 1.0.0
XGA_V052_CONFIG = """
[XGA_SETUP]
xga_save_path = {output}
num_cores = {cores}

[XMM_FILES]
root_xmm_dir = {xmm_pointed_dir}
clean_pn_evts = {{obs_id}}/events/obsid{{obs_id}}-instPN-subexpALL-en-finalevents.fits
clean_mos1_evts = {{obs_id}}/obsid{{obs_id}}-instM1-subexpALL-en-finalevents.fits
clean_mos2_evts = {{obs_id}}/obsid{{obs_id}}-instM2-subexpALL-en-finalevents.fits
attitude_file = {{obs_id}}/P{{obs_id}}OBX000ATTTSR0000.FIT
lo_en = ['0.50', '2.00']
hi_en = ['2.00', '10.00']
pn_image = {{obs_id}}/images/obsid{{obs_id}}-instPN-subexpALL-en{{lo_en}}_{{hi_en}}keV-image.fits
mos1_image = {{obs_id}}/images/obsid{{obs_id}}-instM1-subexpALL-en{{lo_en}}_{{hi_en}}keV-image.fits
mos2_image = {{obs_id}}/images/obsid{{obs_id}}-instM2-subexpALL-en{{lo_en}}_{{hi_en}}keV-image.fits
pn_expmap = {{obs_id}}/images/obsid{{obs_id}}-instPN-subexpALL-en{{lo_en}}_{{hi_en}}keV-expmap.fits
mos1_expmap = {{obs_id}}/images/obsid{{obs_id}}-instM1-subexpALL-en{{lo_en}}_{{hi_en}}keV-expmap.fits
mos2_expmap = {{obs_id}}/images/obsid{{obs_id}}-instM2-subexpALL-en{{lo_en}}_{{hi_en}}keV-expmap.fits
region_file = {xmm_pointed_regs}
                  """

# Config file for xga >= 1.0.0
XGA_V1_CONFIG = XGA_V052_CONFIG + """

[EROSITA_FILES]
root_erosita_dir = {erosita_all_sky_de_dr1_dir}
clean_erosita_evts = {{obs_id}}/events/obsid{{obs_id}}-instTM1_TM2_TM3_TM4_TM5_TM6_TM7-subexpALL-en0.2_10.0keV-finalevents.fits
lo_en = ['0.5']
hi_en = ['2.0']
erosita_image = /this/is/optional/
erosita_expmap = /this/is/optional/
region_file = {erosita_all_sky_de_dr1_regs}
                """

# Collecting information about which missions are compatible with which version of xga, and which
# config files correspond to this version
XGA_VER_DICT = {'0.5.2': {'missions': ['xmm_pointed'],
                           'config_file': XGA_V052_CONFIG},
                 '1.0.0': {'missions': ['erosita_all_sky_de_dr1', 'xmm_pointed'],
                           'config_file': XGA_V1_CONFIG}}

def dict_search(key: str, var: dict) -> list:
    """
    This simple function was very lightly modified from a stackoverflow answer, and is an
    efficient method of searching through a nested dictionary structure for specfic keys
    (and yielding the values associated with them). In this case will extract all of a
    specific product type for a given source.

    :param key: The key in the dictionary to search for and extract values.
    :param var: The variable to search, likely to be either a dictionary or a string.
    :return list[list]: Returns information on keys and values
    """

    # Check that the input is actually a dictionary
    if hasattr(var, 'items'):
        for k, v in var.items():
            if k == key:
                yield v
            # Here is where we dive deeper, recursively searching lower dictionary levels.
            if isinstance(v, dict):
                for result in dict_search(key, v):
                    # We yield a string of the result and the key, as we'll need to return the
                    # ObsID and Instrument information from these product searches as well.
                    # This will mean the output is an unpleasantly nested list, but we can solve that.
                    yield [str(k), result]

def _find_best_version(user_version: str, XGA_VER_DICT: dict, use_latest: bool = False) -> str:
    """
    Internal function used within the write_xga_config function. This parses the user input to the
    "version" argmuent and returns the key in XGA_VER_DICT that is the approriate xga version. E.g.
    if the user parses in '0.6.0', and the available xga versions are '0.5.2' and '1.0.0', this
    function would return '0.5.2'. Can also run this function with use_latest=True, in that case the
    latest version in XGA_VER_DICT is returned.

    :param str user_version: The user input to the 'version' argument in the write_xga_config 
        function.
    :param dict XGA_VER_DICT: XGA_VER_DICT that has keys of xga versions where new telescopes have
        been added.
    :param bool use_latest: If True, the latest xga version in XGA_VER_DICT is returned
    
    :return: The string corresponding to the key in XGA_VER_DICT which best matches the user's 
        input.
    :rtype: str
    """
    # Loading the keys which are strings into a Version class, so they can be compared to other
    # versions more easily
    parsed_versions = {k: Version(k) for k in XGA_VER_DICT.keys()}

    # Sort versions in descending order (latest first)
    sorted_versions = sorted(parsed_versions.items(), key=lambda x: x[1], reverse=True)

    if use_latest:
        return sorted_versions[0]

    else:
        user_parsed = Version(user_version)

        # Find the best match: the highest version ≤ user input
        for key, parsed in sorted_versions:
            if parsed <= user_parsed:
                return key