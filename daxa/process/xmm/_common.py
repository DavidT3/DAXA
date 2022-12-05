#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 05/12/2022, 16:36. Copyright (c) The Contributors
from warnings import warn

from daxa.archive.base import Archive
from daxa.exceptions import NoXMMMissionsError
from daxa.process._backend_check import find_sas

ALLOWED_XMM_MISSIONS = ['xmm_pointed']


def _sas_process_setup(obs_archive: Archive):
    """
    This function is to be called at the beginning of XMM specific processing functions, and contains several
    checks to ensure that passed data common to multiple process function calls is suitable.

    :param Archive obs_archive: The observation archive passed to the processing function that called this function.
    :return:
    :rtype:
    """
    # This makes sure that SAS is installed on the host system, and also identifies the version
    sas_vers = find_sas()

    if not isinstance(obs_archive, Archive):
        raise TypeError('The passed obs_archive must be an instance of the Archive class, which is made up of one '
                        'or more mission class instances.')

    # Now we ensure that the passed observation archive actually contains XMM mission(s)
    xmm_miss = [mission for mission in obs_archive if mission.name in ALLOWED_XMM_MISSIONS]
    if len(xmm_miss) == 0:
        raise NoXMMMissionsError("None of the missions that make up the passed observation archive are "
                                 "XMM missions, and thus this XMM-specific function cannot continue.")
    else:
        processed = [xm.processed for xm in xmm_miss]
        if any(processed):
            warn("One or more XMM missions have already been fully processed")

    return sas_vers


def sas_call(sas_func):
    """
    This is used as a decorator for functions that produce SAS command strings.
    """
    # This is a horrible bodge to make Pycharm not remove SAS_AVAIL and SAS_VERSION from import when it cleans
    #  up prior to committing.
    new_sas_avail = SAS_AVAIL
    new_sas_version = SAS_VERSION

    @wraps(sas_func)
    def wrapper(*args, **kwargs):
        # This has to be here to let autodoc do its noble work without falling foul of these errors
        if not new_sas_avail and new_sas_version is None:
            raise SASNotFoundError("No SAS installation has been found on this machine")
        elif not new_sas_avail:
            raise SASNotFoundError(
                "A SAS installation (v{}) has been found, but the SAS_CCFPATH environment variable is"
                " not set.".format(new_sas_version))

        # The first argument of all of these SAS functions will be the source object (or a list of),
        # so rather than return them from the sas function I'll just access them like this.
        if isinstance(args[0], (BaseSource, NullSource)):
            sources = [args[0]]
        elif isinstance(args[0], (BaseSample, list)):
            sources = args[0]
        else:
            raise TypeError("Please pass a source, NullSource, or sample object.")

        # This is the output from whatever function this is a decorator for
        cmd_list, to_stack, to_execute, cores, p_type, paths, extra_info, disable = sas_func(*args, **kwargs)

        src_lookup = {}
        all_run = []  # Combined command list for all sources
        all_type = []  # Combined expected type list for all sources
        all_path = []  # Combined expected path list for all sources
        all_extras = []  # Combined extra information list for all sources
        source_rep = []  # For repr calls of each source object, needed for assigning products to sources
        for ind in range(len(cmd_list)):
            source = sources[ind]
            if len(cmd_list[ind]) > 0:
                src_lookup[repr(source)] = ind
                # If there are commands to add to a source queue, then do it
                source.update_queue(cmd_list[ind], p_type[ind], paths[ind], extra_info[ind], to_stack)

            # If we do want to execute the commands this time round, we read them out for all sources
            # and add them to these master lists
            if to_execute:
                to_run, expected_type, expected_path, extras = source.get_queue()
                all_run += to_run
                all_type += expected_type
                all_path += expected_path
                all_extras += extras
                source_rep += [repr(source)] * len(to_run)

        # This is what the returned products get stored in before they're assigned to sources
        results = {s: [] for s in src_lookup}
        # Any errors raised shouldn't be SAS, as they are stored within the product object.
        raised_errors = []
        # Making sure something is defined for this variable
        prod_type_str = ""
        if to_execute and len(all_run) > 0:
            # Will run the commands locally in a pool
            prod_type_str = ", ".join(set(all_type))
            with tqdm(total=len(all_run), desc="Generating products of type(s) " + prod_type_str,
                      disable=disable) as gen, Pool(cores) as pool:
                def callback(results_in: Tuple[BaseProduct, str]):
                    """
                    Callback function for the apply_async pool method, gets called when a task finishes
                    and something is returned.
                    :param Tuple[BaseProduct, str] results_in: Results of the command call.
                    """
                    nonlocal gen  # The progress bar will need updating
                    nonlocal results  # The dictionary the command call results are added to
                    if results_in[0] is None:
                        gen.update(1)
                        return
                    else:
                        prod_obj, rel_src = results_in
                        results[rel_src].append(prod_obj)
                        gen.update(1)

                def err_callback(err):
                    """
                    The callback function for errors that occur inside a task running in the pool.
                    :param err: An error that occurred inside a task.
                    """
                    nonlocal raised_errors
                    nonlocal gen

                    if err is not None:
                        # Rather than throwing an error straight away I append them all to a list for later.
                        raised_errors.append(err)
                    gen.update(1)

                for cmd_ind, cmd in enumerate(all_run):
                    # These are just the relevant entries in all these lists for the current command
                    # Just defined like this to save on line length for apply_async call.
                    exp_type = all_type[cmd_ind]
                    exp_path = all_path[cmd_ind]
                    ext = all_extras[cmd_ind]
                    src = source_rep[cmd_ind]
                    pool.apply_async(execute_cmd, args=(str(cmd), str(exp_type), exp_path, ext, src),
                                     error_callback=err_callback, callback=callback)
                pool.close()  # No more tasks can be added to the pool
                pool.join()  # Joins the pool, the code will only move on once the pool is empty.

        elif to_execute and len(all_run) == 0:
            # It is possible to call a wrapped SAS function and find that the products already exist.
            # print("All requested products already exist")
            pass

        # Now we assign products to source objects
        all_to_raise = []
        # This is for the special case of generating an AnnularSpectra product
        ann_spec_comps = {k: [] for k in results}
        for entry in results:
            # Made this lookup list earlier, using string representations of source objects.
            # Finds the ind of the list of sources that we should add this set of products to
            ind = src_lookup[entry]
            to_raise = []
            for product in results[entry]:
                product: BaseProduct
                ext_info = " {s} is the associated source, the specific data used is " \
                           "{o}-{i}.".format(s=sources[ind].name, o=product.obs_id, i=product.instrument)
                if len(product.sas_errors) == 1:
                    to_raise.append(SASGenerationError(product.sas_errors[0] + ext_info))
                elif len(product.sas_errors) > 1:
                    errs = [SASGenerationError(e + ext_info) for e in product.sas_errors]
                    to_raise += errs

                if len(product.errors) == 1:
                    to_raise.append(SASGenerationError(product.errors[0] + "-" + ext_info))
                elif len(product.errors) > 1:
                    errs = [SASGenerationError(e + "-" + ext_info) for e in product.errors]
                    to_raise += errs

                # ccfs aren't actually stored in the source product storage, but they are briefly put into
                #  BaseProducts for error parsing etc. So if the product type is None we don't store it
                if product.type is not None and product.usable and prod_type_str != "annular spectrum set components":
                    # For each product produced for this source, we add it to the storage hierarchy
                    sources[ind].update_products(product)
                elif product.type is not None and product.usable and prod_type_str == "annular spectrum set components":
                    # Really we're just re-creating the results dictionary here, but I want these products
                    #  to go through the error checking stuff like everything else does
                    ann_spec_comps[entry].append(product)

            if len(to_raise) != 0:
                all_to_raise.append(to_raise)

        if prod_type_str == "annular spectrum set components":
            for entry in ann_spec_comps:
                # So now we pass the list of spectra to a AnnularSpectra definition - and it will sort them out
                #  itself so the order doesn't matter
                ann_spec = AnnularSpectra(ann_spec_comps[entry])
                # Refresh the value of ind so that the correct source is used for radii conversion and so that
                #  the AnnularSpectra is added to the correct source.
                ind = src_lookup[entry]
                if sources[ind].redshift is not None:
                    # If we know the redshift we will add the radii to the annular spectra in proper distance units
                    ann_spec.proper_radii = sources[ind].convert_radius(ann_spec.radii, 'kpc')
                # And adding our exciting new set of annular spectra into the storage structure
                sources[ind].update_products(ann_spec)

        # Errors raised here should not be to do with SAS generation problems, but other purely pythonic errors
        for error in raised_errors:
            raise error

        # And here are all the errors during SAS generation, if any
        if len(all_to_raise) != 0:
            raise SASGenerationError(all_to_raise)

        # If only one source was passed, turn it back into a source object rather than a source
        # object in a list.
        if len(sources) == 1:
            sources = sources[0]
        return sources
    return wrapper