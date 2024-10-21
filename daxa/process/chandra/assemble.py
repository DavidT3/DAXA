#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 20/10/2024, 20:16. Copyright (c) The Contributors

from astropy.units import Quantity

from daxa import NUM_CORES
from daxa.archive import Archive
from daxa.process.chandra._common import ciao_call, _ciao_process_setup


@ciao_call
def chandra_repro(obs_archive: Archive, destreak: bool = True, check_very_faint: bool = False, pix_adj: str = 'default',
                  asol_update: bool = True, grating_pi_filter: bool = True, num_cores: int = NUM_CORES,
                  disable_progress: bool = False, timeout: Quantity = None):
    """

    NOTES:
    - We will always produce new bad-pixel files, as we don't currently allow the user to pass their own
    - We will also always produce a new event file with 'process_events', so we don't allow any choice. We want a new
      calibrated level 1 (and from there level 2) event files in all cases.
    - QUESTION FOR ME - set_ardlib controls whether the observation bad-pixel file is stored in the ardlib and (I
      think) doesn't have to be supplied to other analyses. This won't play well with multi-processing I think, and I
      do remember they mentioned ARDLIB in the multi-processing part of their docs. I'll either have to do it that way
      or see whether the bad pixel file can be passed manually for each analysis we might do (which honestly is
      probably the way because XGA is gonna be doing all that stuff). 'set_ardlib=FALSE' FOR NOW!!
    - Following CIAO docs advice for the 'check_vf_pha' parameter and having it set to False by
      default (https://cxc.harvard.edu/ciao/why/aciscleanvf.html.)
    - Don't fully understand 'pix_adj' yet, but should probably include if I don't understand it well enough to not
      be able to argue against it
    - 'tg_zo_position' - this defines the coordinate of the target to be reduced in a grating observation, the 'zero
      point' - we run into the same problem as processing RGS that we just want to prep the data without making
      spectra, ARF, RMF, etc. - as this tool just prepares the data, and the user might not want to ultimately look
      at the brightest source in the field. NOT SURE WHAT TO DO YET - STARTED BY LEAVING IT ON THE DEFAULT BEHAVIOUR
    - 'asol_update' - again don't know why you wouldn't want to do this, but we'll leave the choice in
    - 'pi_filter' - for the grating spectra, a low-cost way of lowering the background it seems? I'll leave the
      choice and it'll be on by default
    - I will initially set verbose to 5, to store the maximum amount of data for debugging

    :param Archive obs_archive:
    :param bool destreak:
    :param bool check_very_faint:
    :param str pix_adj:
    :param bool asol_update:
    :param bool grating_pi_filter:
    :param int num_cores:
    :param bool disable_progress:
    :param Quantity timeout:
    """
    # Runs standard checks, makes directories, returns CIAO versions, etc.
    ciao_vers, caldb_vers, chan_miss = _ciao_process_setup(obs_archive)

    # We're not going to use the handy CIAO Python-wrapped version of 'chandra_repro' - just because all the
    #  infrastructure of DAXA is set up to wrap cmd-line tools itself, and this way we don't rely on there being
    #  a correctly-installed-to-Python version of CIAO (even though that should be quite simple). We are
    #  going to use the chandra_repro command to process Chandra data though, as it works so well, so we
    #  create the command template.
    # Note that we don't need to make a local copy of PFILES here, because that will be added in when the command
    #  is run
    crp_cmd = ("cd {d}; chandra_repro indir={in_f} outdir={out_f} badpixel='yes' process_events='yes' destreak={ds} "
               "set_ardlib='no' check_vf_pha={cvf} pix_adj={pa} tf_zo_position='evt2' asol_update={as_up} "
               "pi_filter={pf} cleanup='no' verbose=5;")
    # "mv *MIEVLI*.FIT ../; mv *ATTTSR*.FIT ../; cd ..; rm -r {d}; mv {oge} {fe}"

    # The 'pix_adj' argument is not boolean, it has several allowed string values, so we check that the user has
    #  not passed something daft before we blindly pass it on the tool command
    if pix_adj not in ['default', 'edser', 'none', 'randomize']:
        raise ValueError("'pix_adj' must be either; 'default', 'edser', 'none', or 'randomize'.")











