#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 22/04/2024, 15:23. Copyright (c) The Contributors
from shutil import copyfile

from tqdm import tqdm

from daxa.archive import Archive
from .setup import create_dirs
from ...exceptions import PreProcessedNotSupportedError


def preprocessed_in_archive(arch: Archive):
    """
    This function acts on an archive's missions which were created with pre-processed data (with things like
    pre-generated event lists, images, and exposure maps downloaded when the archive was set up). It will take the
    existing products and re-organise/re-name them into DAXA's processed archive structure, with the DAXA file
    naming scheme.

    :param Archive arch: A DAXA archive that contains at least one mission with pre-processed data.
    """
    # This is a very inelegant piece of code - but beautiful in function!

    # This will iterate through all the missions associated with the passed archive which have pre-processed data, and
    #  if there are none a suitable error will be raised.
    for miss in arch.preprocessed_missions:
        # Very first thing we want to do is to create the directories in which we will be storing the pre-processed
        #  data - this will do just that (and make a 'failed_data' directory as well, in case any of our pre-processed
        #  data is broken for some reason).
        create_dirs(arch, miss.name)

        # Now we attempt to relocate the products, renaming to our convention
        cur_evt_success = {oi: {} for oi in miss.filtered_obs_ids}
        evt_file_temp = "events/obsid{oi}-inst{i}-subexp{se}-events.fits"
        img_file_temp = "images/obsid{oi}-inst{i}-subexp{se}-en{l}_{h}keV-image.fits"
        exp_file_temp = "images/obsid{oi}-inst{i}-subexp{se}-en{l}_{h}keV-expmap.fits"

        with tqdm(desc="Including pre-processed {pn} data in the archive".format(pn=miss.pretty_name),
                  total=len(miss)) as onwards:
            for obs_id in miss.filtered_obs_ids:
                if miss.name in ['erosita_all_sky_de_dr1', 'erosita_calpv']:
                    if miss.name == "erosita_calpv":
                        rel_act_insts = miss.all_obs_info[miss.all_obs_info['ObsID'] == obs_id].iloc[0]['active_insts']
                    else:
                        rel_act_insts = "TM1,TM2,TM3,TM4,TM5,TM6,TM7"

                    insts = "_".join([i for i in miss.chosen_instruments if i in rel_act_insts])
                    new_name = evt_file_temp.format(oi=obs_id, i=insts, se=None)
                    new_evt_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                    og_evt_path = miss.get_evt_list_path(obs_id)
                    copyfile(og_evt_path, new_evt_path)
                    cur_evt_success[obs_id] = {i: True for i in miss.chosen_instruments if i in rel_act_insts}

                elif not miss.one_inst_per_obs:
                    for inst in miss.chosen_instruments:
                        # TODO Change the se entry when possible
                        new_name = evt_file_temp.format(oi=obs_id, i=inst, se=None)
                        new_evt_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                        try:
                            og_evt_path = miss.get_evt_list_path(obs_id, inst)
                            copyfile(og_evt_path, new_evt_path)
                            cur_evt_success[obs_id][inst] = True
                        except FileNotFoundError:
                            cur_evt_success[obs_id][inst] = False

                else:
                    # All missions with one instrument per ObsID will have an instrument column in their obs info
                    inst = miss.all_obs_info[miss.all_obs_info['ObsID'] == obs_id].iloc[0]['instrument']
                    og_evt_path = miss.get_evt_list_path(obs_id)
                    new_name = evt_file_temp.format(oi=obs_id, i=inst, se=None)
                    new_evt_path = arch.construct_processed_data_path(miss, obs_id) + new_name
                    copyfile(og_evt_path, new_evt_path)
                    cur_evt_success[obs_id][inst] = True

                # If the transfer of event lists was not successful, then nothing else is likely to be
                if not any(cur_evt_success[obs_id].values()):
                    onwards.update(1)
                    continue

                # Again the eROSITA All-Sky data has different rules because it ships with all instruments in one
                #  image/event list/everything
                if miss.name == 'erosita_all_sky_de_dr1':

                    # All the instruments are included
                    insts = 'TM1_TM2_TM3_TM4_TM5_TM6_TM7'

                    # As we know for sure that this mission does have pre-processed energy bands (as this is not
                    #  a general part of this process, but only for eRASS) we just read them out
                    bounds = miss.preprocessed_energy_bands
                    # Grab the bounds for the first of the chosen elements, as they'll all be the same
                    for bnd_pair in bounds[miss.chosen_instruments[0]]:

                        # TODO Change the se entry when possible
                        new_name = img_file_temp.format(oi=obs_id, i=insts, se=None, l=bnd_pair[0].value,
                                                        h=bnd_pair[1].value)
                        new_img_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                        try:
                            og_img_path = miss.get_image_path(obs_id, bnd_pair[0], bnd_pair[1])
                            copyfile(og_img_path, new_img_path)
                        except FileNotFoundError:
                            pass

                        # TODO Change the se entry when possible
                        new_name = exp_file_temp.format(oi=obs_id, i=insts, se=None, l=bnd_pair[0].value,
                                                        h=bnd_pair[1].value)
                        new_exp_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                        try:
                            og_exp_path = miss.get_expmap_path(obs_id, bnd_pair[0], bnd_pair[1])
                            copyfile(og_exp_path, new_exp_path)
                        except FileNotFoundError:
                            pass

                elif miss.name == 'asca':
                    # As we know for sure that this mission does have pre-processed energy bands (as this is not
                    #  a general part of this process, but only for ASCA) we just read them out
                    bounds = miss.preprocessed_energy_bands

                    # ASCA is irritatingly unique in that it ships the images from the two SIS instruments combined,
                    #  and the images from the two GIS instruments combined - thus we have two iterations, one for
                    #  the combined SIS and one for the combined GIS - we need a full identifier though (e.g. SIS1)
                    insts = []
                    for i in miss.chosen_instruments:
                        if i[:-1] not in [s_i[:-1] for s_i in insts]:
                            insts.append(i)
                    for inst in insts:
                        # Grab the bounds for the first of the chosen elements, as they'll all be the same
                        for bnd_pair in bounds[inst]:
                            new_name = img_file_temp.format(oi=obs_id, i=inst, se=None, l=bnd_pair[0].value,
                                                            h=bnd_pair[1].value)
                            new_img_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                            try:
                                og_img_path = miss.get_image_path(obs_id, bnd_pair[0], bnd_pair[1], inst)
                                copyfile(og_img_path, new_img_path)
                            except FileNotFoundError:
                                pass

                            new_name = exp_file_temp.format(oi=obs_id, i=inst, se=None, l=bnd_pair[0].value,
                                                            h=bnd_pair[1].value)
                            new_exp_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                            try:
                                og_exp_path = miss.get_expmap_path(obs_id, bnd_pair[0], bnd_pair[1], inst)
                                copyfile(og_exp_path, new_exp_path)
                            except FileNotFoundError:
                                pass

                elif not miss.one_inst_per_obs:
                    for inst in miss.chosen_instruments:
                        try:
                            bounds = miss.preprocessed_energy_bands
                        except PreProcessedNotSupportedError:
                            onwards.update(1)
                            break

                        for bnd_pair in bounds[inst]:

                            # TODO Change the se entry when possible
                            new_name = img_file_temp.format(oi=obs_id, i=inst, se=None, l=bnd_pair[0].value,
                                                            h=bnd_pair[1].value)
                            new_img_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                            try:
                                og_img_path = miss.get_image_path(obs_id, bnd_pair[0], bnd_pair[1], inst)
                                copyfile(og_img_path, new_img_path)
                            except FileNotFoundError:
                                pass

                            # TODO Change the se entry when possible
                            new_name = exp_file_temp.format(oi=obs_id, i=inst, se=None, l=bnd_pair[0].value,
                                                            h=bnd_pair[1].value)
                            new_exp_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                            try:
                                og_exp_path = miss.get_expmap_path(obs_id, bnd_pair[0], bnd_pair[1], inst)
                                copyfile(og_exp_path, new_exp_path)
                            except FileNotFoundError:
                                pass

                else:
                    # All missions with one instrument per ObsID will have an instrument column in their obs info
                    inst = miss.all_obs_info[miss.all_obs_info['ObsID'] == obs_id].iloc[0]['instrument']

                    try:
                        bounds = miss.preprocessed_energy_bands
                    except PreProcessedNotSupportedError:
                        onwards.update(1)
                        break

                    for bnd_pair in bounds[inst]:
                        # TODO Change the se entry when possible
                        new_name = img_file_temp.format(oi=obs_id, i=inst, se=None, l=bnd_pair[0].value,
                                                        h=bnd_pair[1].value)
                        new_img_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                        try:
                            og_img_path = miss.get_image_path(obs_id, bnd_pair[0], bnd_pair[1], inst)
                            copyfile(og_img_path, new_img_path)
                        except FileNotFoundError:
                            pass

                        new_name = exp_file_temp.format(oi=obs_id, i=inst, se=None, l=bnd_pair[0].value,
                                                        h=bnd_pair[1].value)
                        new_exp_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                        try:
                            og_exp_path = miss.get_expmap_path(obs_id, bnd_pair[0], bnd_pair[1], inst)
                            copyfile(og_exp_path, new_exp_path)
                        except FileNotFoundError:
                            pass

                # TODO CANNOT STAY LIKE THIS AS WILL OVER-UPDATE ONES LIKE eROSITACALPV
                onwards.update(1)
