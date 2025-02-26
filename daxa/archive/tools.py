from packaging.version import Version
from collections import defaultdict
from typing import Union, List

from .. import NUM_CORES
from ..mission import MISS_INDEX
from .base import Archive
from ..mission import Chandra, eRASS1DE, XMMPointed

# Defining XGA information here, I have written the write_xga_config function so that when new
# versions of XGA are released with more telescopes, developers only have to add the new config file
# below, and append this to the XGA_MISS_DICT

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
XGA_MISS_DICT = {'0.5.2': {'missions': ['xmm_pointed'],
                           'config_file': XGA_V052_CONFIG},
                 '1.0.0': {'missions': ['erosita_all_sky_de_dr1', 'xmm_pointed'],
                           'config_file': XGA_V1_CONFIG}}


def _find_best_version(user_version: str, XGA_MISS_DICT: dict, use_latest: bool = False) -> str:
    """
    Internal function used within the write_xga_config function. This parses the user input to the
    "version" argmuent and returns the key in XGA_MISS_DICT that is the approriate xga version. E.g.
    if the user parses in '0.6.0', and the available xga versions are '0.5.2' and '1.0.0', this
    function would return '0.5.2'. Can also run this function with use_latest=True, in that case the
    latest version in XGA_MISS_DICT is returned.

    :param str user_version: The user input to the 'version' argument in the write_xga_config 
        function.
    :param dict XGA_MISS_DICT: XGA_MISS_DICT that has keys of xga versions where new telescopes have
        been added.
    :param bool use_latest: If True, the latest xga version in XGA_MISS_DICT is returned
    
    :return: The string corresponding to the key in XGA_MISS_DICT which best matches the user's 
        input.
    :rtype: str
    """
    # Loading the keys which are strings into a Version class, so they can be compared to other
    # versions more easily
    parsed_versions = {k: Version(k) for k in XGA_MISS_DICT.keys()}

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


def write_xga_config(arch: Archive, xga_version: str = 'latest', 
                     xga_output_path: str = 'xga_output', 
                     num_cores: int = NUM_CORES, include_missions: Union[str, List] = None):
    """
    From an Archive object write an xga config file formatted with paths to the processed data in 
    the Archive.

    :param Archive arch: The Archive object for the config file to be written for.
    :param str xga_version: The xga version to write the config file for. Note that different
        versions of xga have different telescopes that are compatible with it. By default this 
        function writes a file for the latest version of xga.
    :param str xga_output_path: The path in the config file describing where the xga_output
        directory will be written. By default this is left as 'xga_output' which means the output
        of xga is written into the current working directory.
    :param int num_cores: The default number of cores xga can use. By default this function uses the
        number of cores daxa uses.
    :param include_missions: To only include certain missions from an Archive in the config file,
        list the mission names to be included in this argument. By default this function will use 
        all mission in the archive that are compatible with the chosen xga version.
    """
    if not isinstance(arch, Archive):
        raise ValueError("Please parse an Archive object into this argument.")
    
    # From the user's input to the version argument, we want to get the key in XGA_MISS_DICT that
    # best matches it
    if xga_version != 'latest':
        version_key = _find_best_version(xga_version, XGA_MISS_DICT)
    else:
        version_key = _find_best_version(xga_version, XGA_MISS_DICT, use_latest=True)
    
    # From this version key, we can then select the allowed missions and appropriate config file
    # for the chosen xga version
    allowed_missions = XGA_MISS_DICT[version_key]['missions']
    config_file = XGA_MISS_DICT[version_key]['config_file']

    # Checking that the user's input into the include_missions argument is valid
    if include_missions is not None:
        if isinstance(include_missions, str):
            if not include_missions in allowed_missions:
                raise ValueError("The mission given in the include_missions argument is "
                                 "incompatible with the version of xga parsed to this function.")
            else:
                # Parsing this into a list so it can be dealt with consistently later in the func
                include_missions = [include_missions]

        elif isinstance(include_missions, list):
            if not all(isinstance(miss, str) for miss in include_missions):
                raise ValueError("Please parse a list of strings of mission names into the "
                                 "include_missions argument.")
            if not all(miss in allowed_missions for miss in include_missions):
                raise ValueError("Some missions in the include_missions argument are not "
                                 "compatible with the version of xga parsed to this function.")
        else:
            raise ValueError("include_missions must be either a string or a list of strings of "
                             "mission names.")
    
    else:
        include_missions = allowed_missions
    
    # parse missions in the user's archive, only want to keep missions that are allowed and that the
    # user has requested in the include_missions argument
    usable_missions = [miss for miss in arch.mission_names if miss in include_missions]

    if len(usable_missions) == 0:
        raise ValueError("Archive parsed to this function has no missions compatible with XGA.")
    
    # This dict will store the strings that will be input into the config file
    strings_for_config = {}
    for mission in usable_missions:
        strings_for_config[f'{mission}_dir'] = arch.construct_processed_data_path(mission=mission).removesuffix('{oi}/')
        strings_for_config[f'{mission}_regs'] = arch.get_region_file_path(mission=mission)
    
    # We also add the user's choice for xga_output and num_cores
    strings_for_config['output'] = xga_output_path
    strings_for_config['cores'] = num_cores

    # We then turn this dictionary into a defaultdict. When defaultdicts are queried for keys that
    # dont exist, it returns a default value. We use this here, so that when we format the config
    # file with this defaultdict, if there is a telescope that isn't in the user's archive, but is
    # in the config, it will fill those strings with a default value: 'this/is/optional'
    strs_for_cfg_inc_defaults = defaultdict(lambda: '/this/is/optional/', strings_for_config)
    formatted_config = config_file.format_map(strs_for_cfg_inc_defaults)

    # Finally we write the config file
    cfg_dir = arch.archive_path
    cfg_path = cfg_dir + "xga.cfg"
    with open(cfg_path, "w") as cfg:
        cfg.write(formatted_config)
    
    print(f"Wrote config to {cfg_path}")


