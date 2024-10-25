#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 24/10/2024, 20:08. Copyright (c) The Contributors

import os
from random import randint
from typing import Union

import numpy as np
from astropy.units import Quantity, UnitConversionError

from daxa import NUM_CORES
from daxa.archive import Archive
from daxa.exceptions import NoDependencyProcessError
from daxa.process.chandra._common import ciao_call, _ciao_process_setup

# These are the default Chandra Source Catalog (CSC) energy bounds and effective energies (which are used to calculate
#  exposure maps and do the flux conversion to flux maps).
# The 'fluximage' CIAO documentation lists them (https://asc.harvard.edu/ciao/ahelp/fluximage.html#plist.bands), but
#  the choice of effective energies was actually quite involved (see Appendix A
#  of https://cxc.harvard.edu/csc/memos/files/Evans_Requirements.pdf) - hence why I haven't altered the defaults
#  right now
CSC_DEFAULT_EBOUNDS = Quantity([[0.5, 7.0], [0.5, 1.2], [1.2, 2.0], [2.0, 7.0], [0.2, 0.4]], 'keV')
CSC_DEFAULT_EFF_ENERGIES = Quantity([2.3, 0.92, 1.56, 3.8, 0.4], 'keV')


def _internal_flux_image(obs_archive: Archive, mode: str = 'flux', en_bounds: Quantity = CSC_DEFAULT_EBOUNDS,
                         effective_ens: Quantity = CSC_DEFAULT_EFF_ENERGIES, acis_bin_size: Union[float, int] = 4,
                         hrc_bin_size: Union[float, int] = 16, num_cores: int = NUM_CORES,
                         disable_progress: bool = False, timeout: Quantity = None):
    """
    Internal function that does all the heavy lifting to generate Chandra images, exposure maps, and flux (or rate,
    depending on 'mode') maps from processed and cleaned event lists. The 'mode' parameter can be set to
    "flux" (equivalent to the default behaviour of flux_image in CIAO, producing flux images with photon/cm^2/s
    units, and weighted exposure maps with cm^2 s ct/photon units), or "rate" (which produces rate images with
    count/s units, and un-weighted exposure maps in units of seconds). The energy bands and spatial binning can
    be controlled, with each run of this function capable of producing a set of products in different energy bands.

    This function does not face the user because I wanted to provide two discrete functions for flux and rate
    products, that way it is possible to run both and have them show up separately in the archive processing
    history with the current design.

    :param Archive obs_archive: An Archive instance containing a Chandra mission instance. This function will fail
        if no Chandra missions are present in the archive.
    :param str mode: Controls whether the function produces flux maps and weighted event lists ("flux"), or rate maps
        and unweighted event lists ("rate") - default if "flux".
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
                   'badpixfile={bpf} units={m} psfecf=1 parallel="no" tmpdir={td} '
                   'cleanup="yes" verbose=4; {mv_cmd}; cd ..; rm -r {d}')

    # HRC strikes again, doesn't need energy bands of course, and wants another file (the dead time corrections)
    hrc_fi_cmd = ('cd {d}; fluximage infile={cef}[EVENTS] outroot={rn} binsize={bs} asolfile={asol} '
                  'badpixfile={bpf} dtffile={dtf} background="default" units={m} psfecf=1 '
                  'parallel="no" tmpdir={td} cleanup="yes" verbose=4; {mv_cmd}; cd ..; rm -r {d}')

    # The output file names - there have to be a few because this does make a bunch of stuff. The main output
    #  is always the 'flux' file - and it is always called that regardless of the mode.
    prod_im_name = "{rn}_{l}-{u}_thresh.img"
    prod_ex_name = "{rn}_{l}-{u}_thresh.expmap"
    prod_flrt_name = "{rn}_{l}-{u}_flux.img"
    prod_psf_name = "{rn}_{l}-{u}_thresh.psfmap"
    # The HRC file name is different, because we don't specify an energy bound - it also doesn't make all the other
    #  stuff which is a real shame (exposure maps would have been particularly nice).
    prod_hrc_flrt_name = "{rn}_wide.img"

    # Final image, exposure map, rate map, and flux map name templates - cover all eventualities in terms of
    #  whether we're running in flux or rate mode
    im_name = "obsid{oi}-inst{i}-subexp{se}-en{en_id}-image.fits"
    ex_name = "obsid{oi}-inst{i}-subexp{se}-en{en_id}-expmap.fits"
    w_ex_name = "obsid{oi}-inst{i}-subexp{se}-en{en_id}-weightedexpmap.fits"
    rt_name = "obsid{oi}-inst{i}-subexp{se}-en{en_id}-ratemap.fits"
    fl_name = "obsid{oi}-inst{i}-subexp{se}-en{en_id}-fluxmap.fits"
    psf_name = "obsid{oi}-inst{i}-subexp{se}-en{en_id}-psfmap.fits"

    # ---------------------------------- Checking and converting user inputs ----------------------------------
    # Firstly, checking if the energy bounds or effective energies have been changed from default without
    #  changing the other to match
    if (((en_bounds != CSC_DEFAULT_EBOUNDS).any() and (effective_ens == CSC_DEFAULT_EFF_ENERGIES).all()) or
            ((en_bounds == CSC_DEFAULT_EBOUNDS).all() and (effective_ens != CSC_DEFAULT_EFF_ENERGIES).any())):
        raise ValueError("Either the 'en_bounds' or 'effective_ens' argument has been altered from default, without"
                         "changing the other. If one is changed the other must also be altered.")

    # Now will do some sanity checks on the inputs, if they have been changed - only need to check if the energy bounds
    #  have changed here, because we know both have been altered as we got past the error above
    if (en_bounds != CSC_DEFAULT_EBOUNDS).any():

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

    # Want to check that a valid 'mode' has been passed
    if mode not in ['flux', 'rate']:
        raise ValueError("'mode' argument must be set to either 'flux' or 'rate'.")
    # Also create the variable values that actually need to be passed to the command
    elif mode == 'flux':
        unit = 'default'
    else:
        unit = 'time'
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
        good_obs_sel = obs_archive.check_dependence_success(miss.name, all_obs, 'cleaned_chandra_evts',
                                                            no_success_error=False)
        good_obs = np.array(all_obs)[good_obs_sel]

        # Have to check that there is something for us to work with here!
        if len(good_obs) == 0:
            raise NoDependencyProcessError("No observations have had successful 'cleaned_chandra_evts' runs, so "
                                           "'flux_image' cannot be run.")

        for obs_info in good_obs:
            # This is the valid id that allows us to retrieve the specific product for this ObsID-Inst-sub-exposure
            #  (though for Chandra the sub-exposure ID matters very VERY rarely) combo
            val_id = ''.join(obs_info)
            # Split out the information in obs_info
            obs_id, inst, exp_id = obs_info

            # We will need the final cleaned event list - retrieve the path
            rel_evt = obs_archive.process_extra_info[miss.name]['cleaned_chandra_evts'][val_id]['cleaned_events']
            # Also need the aspect solution file
            rel_asol = obs_archive.process_extra_info[miss.name]['chandra_repro'][val_id]['asol_file']
            # And the bad-pixel file
            rel_badpix = obs_archive.process_extra_info[miss.name]['chandra_repro'][val_id]['badpix']

            # Finally, for HRC observations we need the path the dead time file
            if inst == 'HRC':
                rel_dtf = obs_archive.process_extra_info[miss.name]['chandra_repro'][val_id]['dead_time_file']
            else:
                rel_dtf = None

            # This path is guaranteed to exist, as it was set up in _ciao_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.construct_processed_data_path(miss, obs_id)

            # Set up a temporary directory to work in (probably not really necessary in this case, but will be
            #  in other processing functions).
            r_id = randint(0, int(1e+8))
            temp_name = "tempdir_{}".format(r_id)
            temp_dir = dest_dir + temp_name + "/"

            # Also make a root prefix for the files output by flux_image, with the random int above and
            #  the ObsID + instrument identifier
            root_prefix = val_id + "_" + str(r_id)

            # ---------------------------- Creating move commands for generated files ----------------------------
            # Slightly different setup for this function - there are going to be a series of generated files, with
            #  the number decided by the energy bounds passed by the user (at least for ACIS)
            if inst == 'ACIS':
                # This will store the string energy-bound-identifiers for all the requested energy ranges
                en_idents = []
                # This dictionary will get filled with the output file paths (as there are a few different products
                #  produced) - the image and psf types are a constant regardless of operating mode
                final_out_files = {'image': [], 'psf': []}
                # This will be appended too and eventually joined into a string to pass to the command to set
                #  the energy boundaries and the effective energies
                en_cmd_str = []
                # This will be populated with ALL the move commands for all the product types for all the energies,
                #  and then will get stuck on the end of the generation command
                mv_cmd = ""
                mv_temp = "mv {oim} {fim}; mv {oex} {fex}; mv {ofl} {ffl}; mv {opsf} {fpsf};"
                for en_ind, en_bnd in enumerate(en_bounds):
                    lo_en, hi_en = en_bnd
                    en_ident = '{l}_{h}keV'.format(l=lo_en.value, h=hi_en.value)
                    en_idents.append(en_ident)

                    # This is the format that the CIAO flux_image tool requires
                    eff_en = effective_ens[en_ind]
                    cur_en_str = "{l}:{h}:{e}".format(l=lo_en.value, h=hi_en.value, e=eff_en.value)
                    en_cmd_str.append(cur_en_str)

                    # Setting up the file paths where we expect to see flux_image has made our products
                    cur_prod_im = prod_im_name.format(rn=root_prefix, l=lo_en.value, u=hi_en.value)
                    cur_prod_ex = prod_ex_name.format(rn=root_prefix, l=lo_en.value, u=hi_en.value)
                    cur_prod_flrt = prod_flrt_name.format(rn=root_prefix, l=lo_en.value, u=hi_en.value)
                    cur_prod_psf = prod_psf_name.format(rn=root_prefix, l=lo_en.value, u=hi_en.value)

                    # Now we set up the final file paths and the move commands, storing the file paths in the out
                    #  files dictionary (which later gets included in the extra_info dictionary). This is
                    #  ugly and probably there is a better way of doing it but I was stressed out of my mind at this
                    #  point so I simply do not care
                    if mode == 'flux':
                        # In flux mode we make weighted (i.e. not just in seconds) exposure maps, and the
                        #  flux images of photons per s per cm^2
                        final_out_files.setdefault('fluxmap', [])
                        final_out_files.setdefault('weighted_expmap', [])

                        final_flrt = fl_name.format(oi=obs_id, i=inst, se=exp_id, en_id=en_ident)
                        final_flrt = os.path.join(dest_dir, 'images', final_flrt)
                        final_ex = w_ex_name.format(oi=obs_id, i=inst, se=exp_id, en_id=en_ident)
                        final_ex = os.path.join(dest_dir, 'images', final_ex)

                        final_out_files['fluxmap'].append(final_flrt)
                        final_out_files['weighted_expmap'].append(final_ex)
                    else:
                        # This mode makes straight exposure maps in seconds, and count-rate maps in ct/s
                        final_out_files.setdefault('ratemap', [])
                        final_out_files.setdefault('expmap', [])

                        final_flrt = rt_name.format(oi=obs_id, i=inst, se=exp_id, en_id=en_ident)
                        final_flrt = os.path.join(dest_dir, 'images', final_flrt)
                        final_ex = ex_name.format(oi=obs_id, i=inst, se=exp_id, en_id=en_ident)
                        final_ex = os.path.join(dest_dir, 'images', final_ex)

                        final_out_files['ratemap'].append(final_flrt)
                        final_out_files['expmap'].append(final_ex)

                    final_im = im_name.format(oi=obs_id, i=inst, se=exp_id, en_id=en_ident)
                    final_im = os.path.join(dest_dir, 'images', final_im)
                    final_psf = psf_name.format(oi=obs_id, i=inst, se=exp_id, en_id=en_ident)
                    final_psf = os.path.join(dest_dir, 'misc', final_psf)

                    final_out_files['image'].append(final_im)
                    final_out_files['psf'].append(final_psf)

                    # Adding to the move command so that this particular energy range products are put in their
                    #  final places in DAXA's directory hierarchy
                    mv_cmd += mv_temp.format(oim=cur_prod_im, fim=final_im, oex=cur_prod_ex, fex=final_ex,
                                             ofl=cur_prod_flrt, ffl=final_flrt, opsf=cur_prod_psf, fpsf=final_psf)

                # Make the final energy bound command by joining all our list entries
                en_cmd_str = ",".join(en_cmd_str)
                # Remove the last ; so we don't have a double semi-colon in the command
                mv_cmd = mv_cmd[:-1]
            else:
                # And the we have HRC, which is made simpler by a single non-configurable energy range, and the
                #  fact that it only makes one product (which is a shame)
                final_out_files = {}
                en_idents = ["0.06_10.0keV"]
                cur_prod_flrt = prod_hrc_flrt_name.format(rn=root_prefix)
                if mode == 'flux':
                    final_flrt = fl_name.format(oi=obs_id, i=inst, se=exp_id, en_id=en_idents[0])
                    final_flrt = os.path.join(dest_dir, 'images', final_flrt)
                    final_out_files['fluxmap'] = [final_flrt]
                else:
                    final_flrt = rt_name.format(oi=obs_id, i=inst, se=exp_id, en_id=en_idents[0])
                    final_flrt = os.path.join(dest_dir, 'images', final_flrt)
                    final_out_files['ratemap'] = [final_flrt]
                mv_cmd = "mv {ofl} {ffl}".format(ofl=cur_prod_flrt, ffl=final_flrt)
            # ----------------------------------------------------------------------------------------------------

            # Depending on the mode the current function will have a different name in the DAXA histories
            if mode == 'flux':
                func_name = 'flux_image'
            else:
                func_name = 'rate_image'

            # If it doesn't already exist then we will create commands to generate it
            if (func_name not in obs_archive.process_success[miss.name] or
                    val_id not in obs_archive.process_success[miss.name][func_name]):
                # Make the temporary directory for processing - this (along with the temporary PFILES that
                #  the execute_cmd function will create) should help avoid any file collisions
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir + "sub_temp/")

                # The different instruments have different commands again
                if inst == 'ACIS':
                    # Fill out the template, and generate the command that we will run through subprocess
                    cmd = acis_fi_cmd.format(d=temp_dir, cef=rel_evt, rn=root_prefix, eb=en_cmd_str, bs=acis_bin_size,
                                             asol=rel_asol, bpf=rel_badpix, m=unit, mv_cmd=mv_cmd,
                                             td=temp_dir + 'sub_temp/')

                # And here we have an energy-averse instrument (HRC)
                else:
                    cmd = hrc_fi_cmd.format(d=temp_dir, cef=rel_evt, rn=root_prefix, bs=hrc_bin_size, dtf=rel_dtf,
                                            asol=rel_asol, bpf=rel_badpix, m=unit, mv_cmd=mv_cmd,
                                            td=temp_dir + 'sub_temp/')

                # Now store the bash command, the path, and extra info in the dictionaries
                miss_cmds[miss.name][val_id] = cmd
                files_to_check = final_out_files['fluxmap'] if mode == 'flux' else final_out_files['ratemap']
                miss_final_paths[miss.name][val_id] = files_to_check
                miss_extras[miss.name][val_id] = {'working_dir': temp_dir, 'en_idents': en_idents}
                miss_extras[miss.name][val_id].update(final_out_files)

    return miss_cmds, miss_final_paths, miss_extras, "process_message", num_cores, disable_progress, timeout


@ciao_call
def flux_image(obs_archive: Archive, en_bounds: Quantity = CSC_DEFAULT_EBOUNDS,
               effective_ens: Quantity = CSC_DEFAULT_EFF_ENERGIES, acis_bin_size: Union[float, int] = 4,
               hrc_bin_size: Union[float, int] = 16, num_cores: int = NUM_CORES, disable_progress: bool = False,
               timeout: Quantity = None):
    """
    This function is used to generate Chandra images, weighted exposure maps, and flux maps from processed and
    cleaned event lists - flux maps have units of photon/cm^2/s, and weighted exposure maps have units of
    cm^2 s ct/photon. PSF radius maps are also produced by this function. The energy bands and spatial binning can
    be controlled, with each run of this function capable of producing a set of products in different energy bands.

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
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the CIAO generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire process, but a timeout for individual
        ObsID-Inst processes.
    """
    int_ret = _internal_flux_image(obs_archive, 'flux', en_bounds, effective_ens, acis_bin_size, hrc_bin_size,
                                   num_cores, disable_progress, timeout)
    int_ret = list(int_ret)
    # This is just used for populating a progress bar during the process run
    int_ret[3] = 'Generating images, weighted exposure & flux & PSF maps'

    return int_ret


@ciao_call
def rate_image(obs_archive: Archive, en_bounds: Quantity = CSC_DEFAULT_EBOUNDS,
               effective_ens: Quantity = CSC_DEFAULT_EFF_ENERGIES, acis_bin_size: Union[float, int] = 4,
               hrc_bin_size: Union[float, int] = 16, num_cores: int = NUM_CORES, disable_progress: bool = False,
               timeout: Quantity = None):
    """
    This function is used to generate Chandra images, exposure maps, and rate maps from processed and
    cleaned event lists - rate maps have units of count/s, and weighted exposure maps have units of
    seconds. PSF radius maps are also produced by this function. The energy bands and spatial binning can
    be controlled, with each run of this function capable of producing a set of products in different energy bands.

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
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the CIAO generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire process, but a timeout for individual
        ObsID-Inst processes.
    """
    int_ret = _internal_flux_image(obs_archive, 'rate', en_bounds, effective_ens, acis_bin_size, hrc_bin_size,
                                   num_cores, disable_progress, timeout)
    int_ret = list(int_ret)
    # This is just used for populating a progress bar during the process run
    int_ret[3] = 'Generating images, exposure & rate & PSF maps'

    return int_ret
