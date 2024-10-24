#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 24/10/2024, 08:59. Copyright (c) The Contributors

from typing import Union

from astropy.units import Quantity, UnitConversionError

from daxa import NUM_CORES
from daxa.archive import Archive
from daxa.process.chandra._common import ciao_call, _ciao_process_setup

# These are the default Chandra Source Catalog (CSC) energy bounds and effective energies (which are used to calculate
#  exposure maps and do the flux conversion to flux maps).
# The 'fluximage' CIAO documentation lists them (https://asc.harvard.edu/ciao/ahelp/fluximage.html#plist.bands), but
#  the choice of effective energies was actually quite involved (see Appendix A
#  of https://cxc.harvard.edu/csc/memos/files/Evans_Requirements.pdf) - hence why I haven't altered the defaults
#  right now
CSC_DEFAULT_EBOUNDS = Quantity([[0.5, 7.0], [0.5, 1.2], [1.2, 2.0], [2.0, 7.0], [0.2, 0.4]], 'keV')
CSC_DEFAULT_EFF_ENERGIES = Quantity([2.3, 0.92, 1.56, 3.8, 0.4], 'keV')


@ciao_call
def flux_image(obs_archive: Archive, mode: str = 'flux', en_bounds: Quantity = CSC_DEFAULT_EBOUNDS,
               effective_ens: Quantity = CSC_DEFAULT_EFF_ENERGIES, acis_bin_size: Union[float, int] = 4,
               hrc_bin_size: Union[float, int] = 16, num_cores: int = NUM_CORES, disable_progress: bool = False,
               timeout: Quantity = None):
    """
    This function is used to generate Chandra images, exposure maps, and flux (or rate, depending on 'mode') maps
    from processed and cleaned event lists. The 'mode' parameter can be set to "flux" (equivalent to the default
    behaviour of flux_image in CIAO, producing flux images with photon/cm^2/s units, and weighted exposure maps with
    cm^2 s ct/photon units), or "rate" (which produces rate images with count/s units, and un-weighted exposure maps in
    units of seconds). The energy bands and spatial binning can be controlled, with each run of this function
    capable of producing a set of products in different energy bands.

    :param Archive obs_archive: An Archive instance containing a Chandra mission instance. This function will fail
        if no Chandra missions are present in the archive.
    :param Quantity en_bounds: The energy bounds in which to generate images, exposure maps, and flux maps/rate
        maps. Should be passed as a 2D array quantity with shape (N, 2), where N is the number of different energy
        bounds, in units convertible to keV. Default are the Chandra Source Catalog (CSC) boundaries - be aware that
        changing the 'en_bounds' parameter will also necessitate changes to the 'effective_ens' parameter. Energy
        bounds are NOT applied to HRC data products.
    :param Quantity effective_ens: The effective energies for the energy bounds set in 'en_bounds' - consider them
        almost as a "central energy" at which exposure maps are calculated. The default values are the Chandra
        Source Catalog (CSC) effective energies (to match the default value of 'en_bounds'). If the 'en_bounds'
        argument is altered, this argument will need to be changed as well.
    :param int/float acis_bin_size: The image binning factor to be applied to ACIS image generation - this decides the
        size of output product pixels, with smaller values resulting in finer binning. The output product pixel size
        for ACIS will be 'acis_bin_size'*0.492 arcseconds. Default is 4 (finer by default than CIAO flux_image).
    :param int/float hrc_bin_size: The image binning factor to be applied to HRC image generation - this decides the
        size of output product pixels, with smaller values resulting in finer binning. The output product pixel size
        for HRC will be 'hrc_bin_size'*0.1318 arcseconds. Default is 16 (finer by default than CIAO flux_image).
    :param str mode:
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the CIAO generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire process, but a timeout for individual
        ObsID-Inst processes.
    """
    # Run the setup for Chandra processes, which checks that CIAO is installed (as well as CALDB), and checks that the
    #  archive has at least one Chandra mission in it, and
    ciao_vers, caldb_vers, chan_miss = _ciao_process_setup(obs_archive)

    #
    acis_fi_cmd = ('cd {d}; fluximage infile={cef}[EVENTS] outroot={rn} bands={eb} binsize={bs} asolfile={asol} '
                   'badpixfile={bpf} units={m} expmapthresh={exth} psfecf="yes" parallel="no" tmpdir={d} '
                   'cleanup="no" verbose=5; ')
    # 'cd ..; rm -r {d}'

    # HRC strikes again, doesn't need energy bands of course, and wants another file (the dead time corrections)
    hrc_fi_cmd = ('cd {d}; fluximage infile={cef}[EVENTS] outroot={rn} binsize={bs} asolfile={asol} '
                  'badpixfile={bpf} dtffile={dtf} background="default" units={m} expmapthresh={exth} psfecf="yes" '
                  'parallel="no" tmpdir={d} cleanup="no" verbose=5; ')

    # TODO ADD DEAD TIME CORRECTION FILE TO HRC COMMAND - WHEN I SAVE IT PROPERLY
    # TODO CONSIDER LETTING USER DECIDE WHETHER TO APPLY PARTICLE BACKGROUND CORRECTION TO HRC-I DATA

    # Final image, exposure map, rate map, and flux map name templates
    im_name = ""
    ex_name = ""
    rt_name = ""
    fl_name = ""

    # ---------------------------------- Checking and converting user inputs ----------------------------------
    # Firstly, checking if the energy bounds or effective energies have been changed from default without
    #  changing the other to match
    if ((en_bounds != CSC_DEFAULT_EBOUNDS and effective_ens == CSC_DEFAULT_EFF_ENERGIES) or
            (en_bounds == CSC_DEFAULT_EBOUNDS and effective_ens != CSC_DEFAULT_EFF_ENERGIES)):
        raise ValueError("Either the 'en_bounds' or 'effective_ens' argument has been altered from default, without"
                         "changing the other. If one is changed the other must also be altered.")

    # Now will do some sanity checks on the inputs, if they have been changed - only need to check if the energy bounds
    #  have changed here, because we know both have been altered as we got past the error above
    if en_bounds != CSC_DEFAULT_EBOUNDS:

        # If they've been altered, want to make sure they are in the expected format
        if en_bounds.isscalar or effective_ens.isscalar:
            raise ValueError("The 'en_bounds' and 'effective_ens' arguments must be arrays, not a scalar quantity.")
        elif en_bounds.ndim != 2 or en_bounds.shape[1] != 2:
            raise ValueError("The 'en_bounds' argument must be an Nx2 array of lower (first column) and upper "
                             "(second column) energy bounds.")
        elif effective_ens.ndim != 1 or len(effective_ens) != en_bounds.shape[0]:
            raise ValueError("The 'effective_ens' argument must be a 1D quantity with the same number of entries "
                             "as there are energy bounds defined by 'en_bounds'.")
        elif not en_bounds.unit.is_equivalent('keV') or not effective_ens.unit.is_equivalent('keV'):
            raise UnitConversionError("The 'en_bounds' and 'effective_ens' arguments must be in units convertible to"
                                      " keV.")

        # Make sure we convert the units to keV
        en_bounds = en_bounds.to('keV')
        effective_ens = effective_ens.to('keV')

        # If we've gotten this far we know the energy bounds and effective energies are in the correct format, now
        #  we'll check the validity of them as far as we can
        if (en_bounds[:, 0] >= en_bounds[:, 1]).any():
            raise ValueError("Lower energy bounds must be less than upper energy bounds.")
        elif (effective_ens < en_bounds[:, 0]).any() or (effective_ens > en_bounds[:, 1]).any():
            raise ValueError("The energies defined in 'effective_ens' must be within their matching energy bounds.")

    # ---------------------------------------------------------------------------------------------------------

    # Sets up storage dictionaries for bash commands, final file paths (to check they exist at the end), and any
    #  extra information
    miss_cmds = {}
    miss_final_paths = {}
    miss_extras = {}

    # We are iterating through Chandra missions, though only one type exists in DAXA and I don't see that changing.
    #  Much of this code is boilerplate that you'll see throughout the Chandra functions (and similar code in many of
    #  the other telescope processing functions), but never mind - it doesn't need to be that different, so why
    #  should we make it so?
    for miss in chan_miss:
        # Sets up the top level keys (mission name) in our storage dictionaries
        miss_cmds[miss.name] = {}
        miss_final_paths[miss.name] = {}
        miss_extras[miss.name] = {}

        # Getting all the ObsIDs that have been flagged as being able to be processed
        all_obs = obs_archive.get_obs_to_process(miss.name)
        # Then filtering those based on which of them successfully passed the dependency function
        good_obs_sel = obs_archive.check_dependence_success(miss.name, all_obs, 'deflare', no_success_error=False)
        good_obs = np.array(all_obs)[good_obs_sel]

        # Have to check that there is something for us to work with here!
        if len(good_obs) == 0:
            raise NoDependencyProcessError("No observations have had successful 'deflare' runs, so "
                                           "'cleaned_chandra_evts' cannot be run.")

        for obs_info in good_obs:
            # This is the valid id that allows us to retrieve the specific product for this ObsID-Inst-sub-exposure
            #  (though for Chandra the sub-exposure ID matters very VERY rarely) combo
            val_id = ''.join(obs_info)
            # Split out the information in obs_info
            obs_id, inst, exp_id = obs_info

            # We will need the event list created by the 'chandra_repro' run, so the path must be retrieved
            rel_evt = obs_archive.process_extra_info[miss.name]['chandra_repro'][val_id]['evt_list']
            # We will also need the flaring GTI produced by 'deflare', and again that is in the process extra info
            rel_flare_gti = obs_archive.process_extra_info[miss.name]['deflare'][val_id]['flaring_gti']

            # This path is guaranteed to exist, as it was set up in _ciao_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.construct_processed_data_path(miss, obs_id)

            # Set up a temporary directory to work in (probably not really necessary in this case, but will be
            #  in other processing functions).
            r_id = randint(0, int(1e+8))
            temp_name = "tempdir_{}".format(r_id)
            temp_dir = dest_dir + temp_name + "/"

            # ------------------------------ Creating final name for output evt file ------------------------------
            # Also need to set up the name for the interim event list, where filtering expressions. Note that this
            #  lives in the temporary directory, and will be lost to the ages when that directory is deleted at
            #  the end of the process.
            int_evt_final_path = os.path.join(temp_dir, int_evt_name.format(o=obs_id, se=exp_id, i=inst,
                                                                            en_id=en_ident))

            # This is where the final 'clean' event list will live, hopefully devoid of flares and unpleasant events
            cl_evt_final_path = os.path.join(dest_dir, 'events', cl_evt_name.format(o=obs_id, se=exp_id, i=inst,
                                                                                    en_id=en_ident))
            # -----------------------------------------------------------------------------------------------------

            # If it doesn't already exist then we will create commands to generate it
            if ('cleaned_chandra_evts' not in obs_archive.process_success[miss.name] or
                    val_id not in obs_archive.process_success[miss.name]['cleaned_chandra_evts']):
                # Make the temporary directory for processing - this (along with the temporary PFILES that
                #  the execute_cmd function will create) should help avoid any file collisions
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                # Slightly different commands based on whether the user wants us to apply an energy cut or not (or
                #  if an energy cut can't be applied - HRC data)
                if lo_en is not None:
                    # Fill out the template, and generate the command that we will run through subprocess
                    cmd = en_clevt_cmd.format(d=temp_dir, ef=rel_evt, lo_en=lo_en.value, hi_en=hi_en.value,
                                              gr=allowed_grades, iev=int_evt_final_path, fgti=rel_flare_gti,
                                              fev=cl_evt_final_path)
                # Here we have no energy cut
                elif lo_en is None and inst != 'HRC':
                    cmd = no_en_clevt_cmd.format(d=temp_dir, ef=rel_evt, gr=allowed_grades, iev=int_evt_final_path,
                                                 fgti=rel_flare_gti, fev=cl_evt_final_path)
                # And here we have an energy-cut-averse instrument (HRC) that also has fundamentally different
                #  information in the event lists
                else:
                    cmd = hrc_clevt_cmd.format(d=temp_dir, ef=rel_evt, iev=int_evt_final_path, fgti=rel_flare_gti,
                                               fev=cl_evt_final_path)

                # Now store the bash command, the path, and extra info in the dictionaries
                miss_cmds[miss.name][val_id] = cmd
                miss_final_paths[miss.name][val_id] = cl_evt_final_path
                miss_extras[miss.name][val_id] = {'working_dir': temp_dir, 'cleaned_events': cl_evt_final_path,
                                                  'en_key': en_ident}

    # This is just used for populating a progress bar during the process run
    process_message = 'Assembling cleaned event lists'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout