#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 11/12/2022, 16:44. Copyright (c) The Contributors

from daxa import NUM_CORES
from daxa.archive.base import Archive
from daxa.process.xmm._common import _sas_process_setup


def epchain():
    pass


def emchain(obs_archive: Archive, num_cores: int = NUM_CORES, disable_progress: bool = False):
    # Run the setup for SAS processes, which checks that SAS is installed, checks that the archive has at least
    #  one XMM mission in it, and shows a warning if the XMM missions have already been processed
    sas_version = _sas_process_setup(obs_archive)

    # Define the form of the odfingest command that must be run to create an ODF summary file
    # odf_cmd = "cd {d}; export SAS_CCF={ccf}; echo $SAS_CCF; odfingest odfdir={odf_dir} outdir={out_dir} withodfdir=yes"
    em_cmd = "cd {d}; export SAS_CCF={ccf}; emchain odf={od}"

    # Sets up storage dictionaries for bash commands, final file paths (to check they exist at the end), and any
    #  extra information that might be useful to provide to the next step in the generation process
    miss_cmds = {}
    miss_final_paths = {}
    miss_extras = {}

