#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 23/11/2022, 19:03. Copyright (c) The Contributors

import os
from configparser import ConfigParser
from warnings import warn

from numpy import floor

from .exceptions import DAXAConfigError

# If XDG_CONFIG_HOME is set, then use that, otherwise use this default config path
CONFIG_PATH = os.environ.get('XDG_CONFIG_HOME', os.path.join(os.path.expanduser('~'), '.config', 'daxa'))
# DAXA config file path
CONFIG_FILE = os.path.join(CONFIG_PATH, 'daxa.cfg')
# Section of the config file for setting up the DAXA module
DAXA_CONFIG = {"daxa_save_path": "daxa_output/",
               "global_archive_database": False,
               "num_cores": -1}

if not os.path.exists(CONFIG_PATH):
    os.makedirs(CONFIG_PATH)

# If first DAXA run, creates default config file
if not os.path.exists(CONFIG_FILE):
    daxa_default = ConfigParser()
    daxa_default.add_section("DAXA_SETUP")
    daxa_default["DAXA_SETUP"] = DAXA_CONFIG
    with open(CONFIG_FILE, 'w') as new_cfg:
        daxa_default.write(new_cfg)

    # First time run triggers this message
    warn("A configuration file has been created ({}); you can use it to control where DAXA "
         "stores data by default.".format(CONFIG_FILE))

daxa_conf = ConfigParser()
# It would be nice to do configparser interpolation, but it wouldn't handle the lists of energy values
daxa_conf.read(CONFIG_FILE)

if not isinstance(daxa_conf["DAXA_SETUP"]["global_archive_database"], bool):
    raise DAXAConfigError("The value for global_archive_database must be either True or False.")

# As it turns out, the ConfigParser class is a pain to work with, so we're converting to a dict here
# Addressing works just the same
daxa_conf = {str(sect): dict(daxa_conf[str(sect)]) for sect in daxa_conf}

if daxa_conf["DAXA_SETUP"]["num_cores"] != -1:
    # If the user has set a number of cores in the config file then we'll use that.
    NUM_CORES = int(daxa_conf["DAXA_SETUP"]["num_cores"])
else:
    # Going to allow multi-core processing to use 90% of available cores by default, but
    # this can be over-ridden in individual SAS calls.
    NUM_CORES = max(int(floor(os.cpu_count() * 0.9)), 1)  # Makes sure that at least one core is used

# This is the default output directory for archives setup by DAXA, though it can be overridden on
#  a mission level
OUTPUT = os.path.abspath(daxa_conf["DAXA_SETUP"]["daxa_save_path"]) + "/"


