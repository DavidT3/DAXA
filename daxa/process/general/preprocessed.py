#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 03/10/2024, 23:00. Copyright (c) The Contributors

from shutil import copyfile
from typing import List

from tqdm import tqdm

from daxa.archive import Archive
from .setup import create_dirs
from ... import BaseMission
from ...exceptions import PreProcessedNotSupportedError


def preprocessed_in_archive(arch: Archive, missions: List[str] = None):
    """
    This function acts on an archive's missions which were created with pre-processed data (with things like
    pre-generated event lists, images, and exposure maps downloaded when the archive was set up). It will take the
    existing products and re-organise/re-name them into DAXA's processed archive structure, with the DAXA file
    naming scheme.

    :param Archive arch: A DAXA archive that contains at least one mission with pre-processed data.
    :param List[BaseMission] missions: Optionally, a list of mission names that are to have their preprocessed data
        reorganised into the DAXA archive. Default is None, in which case all 'pre-processed' missions will be
        acted upon.
    """
    # This is a very inelegant piece of code - but beautiful in function!

    # First of all, check the missions input
    preproc_miss_names = [miss.name for miss in arch.preprocessed_missions]
    if missions is not None and (not isinstance(missions, list) and
                                 all([en in preproc_miss_names for en in missions])):
        raise TypeError("The 'missions' argument must be a list of names of missions associated with the archive that "
                        "have been pre-processed.")

    # Make sure that if no list has been passed then we just use all the preprocessed missions
    if missions is None:
        rel_miss = arch.preprocessed_missions
    else:
        rel_miss = [arch[mn] for mn in missions]

    # This will iterate through all the missions associated with the passed archive which have pre-processed data, and
    #  if there are none a suitable error will be raised.
    evt_success = {}
    img_success = {}
    exp_success = {}
    bck_success = {}

    for miss in rel_miss:
        # Very first thing we want to do is to create the directories in which we will be storing the pre-processed
        #  data - this will do just that (and make a 'failed_data' directory as well, in case any of our pre-processed
        #  data is broken for some reason).
        create_dirs(arch, miss.name)

        # Now we attempt to relocate the products, renaming to our convention
        cur_evt_success = {}
        cur_img_success = {}
        cur_exp_success = {}
        cur_bck_success = {}
        evt_file_temp = "events/obsid{oi}-inst{i}-subexp{se}-finalevents.fits"
        img_file_temp = "images/obsid{oi}-inst{i}-subexp{se}-en{l}_{h}keV-image.fits"
        exp_file_temp = "images/obsid{oi}-inst{i}-subexp{se}-en{l}_{h}keV-expmap.fits"
        bck_file_temp = "background/obsid{oi}-inst{i}-subexp{se}-en{l}_{h}keV-backmap.fits"

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
                    cur_evt_success.update({obs_id+i: True for i in miss.chosen_instruments if i in rel_act_insts})

                elif not miss.one_inst_per_obs:
                    for inst in miss.chosen_instruments:
                        # TODO Change the se entry when possible
                        new_name = evt_file_temp.format(oi=obs_id, i=inst, se=None)
                        new_evt_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                        try:
                            og_evt_path = miss.get_evt_list_path(obs_id, inst)
                            copyfile(og_evt_path, new_evt_path)
                            cur_evt_success[obs_id+inst] = True
                        except FileNotFoundError:
                            cur_evt_success[obs_id+inst] = False
                        except PreProcessedNotSupportedError:
                            pass

                else:
                    # All missions with one instrument per ObsID will have an instrument column in their obs info
                    inst = miss.all_obs_info[miss.all_obs_info['ObsID'] == obs_id].iloc[0]['instrument']
                    og_evt_path = miss.get_evt_list_path(obs_id)
                    new_name = evt_file_temp.format(oi=obs_id, i=inst, se=None)
                    new_evt_path = arch.construct_processed_data_path(miss, obs_id) + new_name
                    copyfile(og_evt_path, new_evt_path)
                    cur_evt_success[obs_id+inst] = True

                # If the transfer of event lists was not successful, then nothing else is likely to be
                if not any([succ for ident, succ in cur_evt_success.items() if obs_id in ident]):
                    onwards.update(1)
                    continue

                # ------------------------------ Images/ExpMaps/BackMaps ---------------------------------------
                # ----------------------------------------------------------------------------------------------
                # Again the eROSITA All-Sky data has different rules because it ships with all instruments in one
                #  image/event list/everything
                if miss.name == 'erosita_all_sky_de_dr1':

                    # All the instruments are included
                    insts = 'TM1_TM2_TM3_TM4_TM5_TM6_TM7'

                    # As we know for sure that this mission does have pre-processed energy bands (as this is not
                    #  a general part of this process, but only for eRASS) we just read them out
                    bounds = miss.preprocessed_energy_bands

                    # This is just the first of the chosen instruments, as they're all lumped together
                    bodge_inst = miss.chosen_instruments[0]
                    # Grab the bounds for the first of the chosen elements, as they'll all be the same
                    for bnd_pair in bounds[bodge_inst]:

                        # TODO Change the se entry when possible
                        new_name = img_file_temp.format(oi=obs_id, i=insts, se=None, l=bnd_pair[0].value,
                                                        h=bnd_pair[1].value)
                        new_img_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                        try:
                            og_img_path = miss.get_image_path(obs_id, bnd_pair[0], bnd_pair[1])
                            copyfile(og_img_path, new_img_path)
                            if bodge_inst+obs_id not in cur_img_success or not cur_img_success[obs_id+bodge_inst]:
                                cur_img_success.update({obs_id+i: True for i in miss.chosen_instruments})
                        except FileNotFoundError:
                            cur_img_success.update({obs_id + i: False for i in miss.chosen_instruments})
                        except PreProcessedNotSupportedError:
                            pass

                        # TODO Change the se entry when possible
                        new_name = exp_file_temp.format(oi=obs_id, i=insts, se=None, l=bnd_pair[0].value,
                                                        h=bnd_pair[1].value)
                        new_exp_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                        try:
                            og_exp_path = miss.get_expmap_path(obs_id, bnd_pair[0], bnd_pair[1])
                            copyfile(og_exp_path, new_exp_path)
                            if obs_id+bodge_inst not in cur_exp_success or not cur_exp_success[obs_id+bodge_inst]:
                                cur_exp_success.update({obs_id+i: True for i in miss.chosen_instruments})
                        except FileNotFoundError:
                            cur_exp_success.update({obs_id+i: False for i in miss.chosen_instruments})
                        except PreProcessedNotSupportedError:
                            pass

                        # TODO Change the se entry when possible
                        new_name = bck_file_temp.format(oi=obs_id, i=insts, se=None, l=bnd_pair[0].value,
                                                        h=bnd_pair[1].value)
                        new_bck_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                        try:
                            og_bck_path = miss.get_background_path(obs_id, bnd_pair[0], bnd_pair[1])
                            copyfile(og_bck_path, new_bck_path)
                            if obs_id+bodge_inst not in cur_bck_success or not cur_bck_success[obs_id+bodge_inst]:
                                cur_bck_success.update({obs_id+i: True for i in miss.chosen_instruments})
                        except FileNotFoundError:
                            cur_bck_success.update({obs_id+i: False for i in miss.chosen_instruments})
                        except PreProcessedNotSupportedError:
                            pass

                    onwards.update(1)

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
                                if obs_id+inst not in cur_img_success or not cur_img_success[obs_id+inst]:
                                    cur_img_success.update({obs_id+i: True for i in miss.chosen_instruments
                                                            if inst[:-1] in i})
                            except FileNotFoundError:
                                cur_img_success.update({obs_id+i: False for i in miss.chosen_instruments
                                                        if inst[:-1] in i})
                            except PreProcessedNotSupportedError:
                                pass

                            new_name = exp_file_temp.format(oi=obs_id, i=inst, se=None, l=bnd_pair[0].value,
                                                            h=bnd_pair[1].value)
                            new_exp_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                            try:
                                og_exp_path = miss.get_expmap_path(obs_id, bnd_pair[0], bnd_pair[1], inst)
                                copyfile(og_exp_path, new_exp_path)
                                if obs_id+inst not in cur_exp_success or not cur_exp_success[obs_id+inst]:
                                    cur_exp_success.update({obs_id+i: True for i in miss.chosen_instruments
                                                            if inst[:-1] in i})
                            except FileNotFoundError:
                                cur_exp_success.update({obs_id+i: False for i in miss.chosen_instruments
                                                        if inst[:-1] in i})
                            except PreProcessedNotSupportedError:
                                pass

                    onwards.update(1)
                elif not miss.one_inst_per_obs:
                    for inst in miss.chosen_instruments:
                        try:
                            bounds = miss.preprocessed_energy_bands
                        except PreProcessedNotSupportedError:
                            break

                        for bnd_pair in bounds[inst]:

                            # TODO Change the se entry when possible
                            new_name = img_file_temp.format(oi=obs_id, i=inst, se=None, l=bnd_pair[0].value,
                                                            h=bnd_pair[1].value)
                            new_img_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                            try:
                                og_img_path = miss.get_image_path(obs_id, bnd_pair[0], bnd_pair[1], inst)
                                copyfile(og_img_path, new_img_path)
                                if obs_id+inst not in cur_img_success or not cur_img_success[obs_id+inst]:
                                    cur_img_success[obs_id+inst] = True
                            except FileNotFoundError:
                                cur_img_success[obs_id+inst] = False
                            except PreProcessedNotSupportedError:
                                pass

                            # TODO Change the se entry when possible
                            new_name = exp_file_temp.format(oi=obs_id, i=inst, se=None, l=bnd_pair[0].value,
                                                            h=bnd_pair[1].value)
                            new_exp_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                            try:
                                og_exp_path = miss.get_expmap_path(obs_id, bnd_pair[0], bnd_pair[1], inst)
                                copyfile(og_exp_path, new_exp_path)
                                if obs_id+inst not in cur_exp_success or not cur_exp_success[obs_id+inst]:
                                    cur_exp_success[obs_id+inst] = True
                            except FileNotFoundError:
                                cur_exp_success[obs_id+inst] = False
                            except PreProcessedNotSupportedError:
                                pass

                            # TODO Change the se entry when possible
                            new_name = bck_file_temp.format(oi=obs_id, i=inst, se=None, l=bnd_pair[0].value,
                                                            h=bnd_pair[1].value)
                            new_bck_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                            try:
                                og_bck_path = miss.get_background_path(obs_id, bnd_pair[0], bnd_pair[1], inst)
                                copyfile(og_bck_path, new_bck_path)
                                if obs_id+inst not in cur_bck_success or not cur_bck_success[obs_id+inst]:
                                    cur_bck_success[obs_id+inst] = True
                            except FileNotFoundError:
                                cur_bck_success[obs_id+inst] = False
                            except PreProcessedNotSupportedError:
                                pass
                    onwards.update(1)
                else:
                    # All missions with one instrument per ObsID will have an instrument column in their obs info
                    inst = miss.all_obs_info[miss.all_obs_info['ObsID'] == obs_id].iloc[0]['instrument']

                    try:
                        bounds = miss.preprocessed_energy_bands
                    except PreProcessedNotSupportedError:
                        continue

                    for bnd_pair in bounds[inst]:
                        # TODO Change the se entry when possible
                        new_name = img_file_temp.format(oi=obs_id, i=inst, se=None, l=bnd_pair[0].value,
                                                        h=bnd_pair[1].value)
                        new_img_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                        try:
                            og_img_path = miss.get_image_path(obs_id, bnd_pair[0], bnd_pair[1], inst)
                            copyfile(og_img_path, new_img_path)
                            if obs_id+inst not in cur_img_success or not cur_img_success[obs_id+inst]:
                                cur_img_success[obs_id+inst] = True
                        except FileNotFoundError:
                            cur_img_success[obs_id+inst] = False
                        except PreProcessedNotSupportedError:
                            pass

                        new_name = exp_file_temp.format(oi=obs_id, i=inst, se=None, l=bnd_pair[0].value,
                                                        h=bnd_pair[1].value)
                        new_exp_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                        try:
                            og_exp_path = miss.get_expmap_path(obs_id, bnd_pair[0], bnd_pair[1], inst)
                            copyfile(og_exp_path, new_exp_path)
                            if obs_id+inst not in cur_exp_success or not cur_exp_success[obs_id+inst]:
                                cur_exp_success[obs_id+inst] = True
                        except FileNotFoundError:
                            cur_exp_success[obs_id+inst] = False
                        except PreProcessedNotSupportedError:
                            pass

                        # TODO Change the se entry when possible
                        new_name = bck_file_temp.format(oi=obs_id, i=inst, se=None, l=bnd_pair[0].value,
                                                        h=bnd_pair[1].value)
                        new_bck_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                        try:
                            og_bck_path = miss.get_background_path(obs_id, bnd_pair[0], bnd_pair[1], inst)
                            copyfile(og_bck_path, new_bck_path)
                            if obs_id+inst not in cur_bck_success or not cur_bck_success[obs_id+inst]:
                                cur_bck_success[obs_id+inst] = True
                        except FileNotFoundError:
                            cur_bck_success[obs_id+inst] = False
                        except PreProcessedNotSupportedError:
                            pass
                    onwards.update(1)

        # TODO NEED TO FIX THESE
        evt_success[miss.name] = cur_evt_success
        if len(cur_img_success) != 0:
            img_success[miss.name] = cur_img_success
        if len(cur_exp_success) != 0:
            exp_success[miss.name] = cur_exp_success
        if len(cur_bck_success) != 0:
            bck_success[miss.name] = cur_bck_success

        # This sets the archive status for this mission to fully processed
        arch[miss.name].processed = True

    arch.process_success = ('preprocessed_events', evt_success)
    arch.process_success = ('preprocessed_images', img_success)
    arch.process_success = ('preprocessed_expmaps', exp_success)
    arch.process_success = ('preprocessed_backmaps', bck_success)
    # Make sure to save the archive at the end of this
    arch.save()
