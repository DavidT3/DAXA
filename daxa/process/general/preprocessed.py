#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 22/04/2024, 13:22. Copyright (c) The Contributors
from shutil import copyfile

from daxa.archive import Archive
from .setup import create_dirs


def preprocessed_in_archive(arch: Archive):
    """
    This function acts on an archive's missions which were created with pre-processed data (with things like
    pre-generated event lists, images, and exposure maps downloaded when the archive was set up). It will take the
    existing products and re-organise/re-name them into DAXA's processed archive structure, with the DAXA file
    naming scheme.

    :param Archive arch: A DAXA archive that contains at least one mission with pre-processed data.
    """

    # This will iterate through all the missions associated with the passed archive which have pre-processed data, and
    #  if there are none a suitable error will be raised.
    for miss in arch.preprocessed_missions:
        # Very first thing we want to do is to create the directories in which we will be storing the pre-processed
        #  data - this will do just that (and make a 'failed_data' directory as well, in case any of our pre-processed
        #  data is broken for some reason).
        create_dirs(arch, miss.name)

        cur_evt_success = {oi: {} for oi in miss.filtered_obs_ids}

        for obs_id in miss.filtered_obs_ids:
            if miss.name in ['erosita_all_sky_de_dr1', 'erosita_calpv']:
                if miss.name == "erosita_calpv":
                    rel_act_insts = miss.all_obs_info[miss.all_obs_info['ObsID'] == obs_id].iloc[0]['active_insts']
                else:
                    rel_act_insts = "TM1,TM2,TM3,TM4,TM5,TM6,TM7"

                insts = "".join([i for i in miss.chosen_instruments if i in rel_act_insts])
                new_name = "obsid{oi}_inst{i}_subexp{se}.fits".format(oi=obs_id, i=insts, se=None)
                new_evt_path = arch.construct_processed_data_path(miss, obs_id) + new_name

                og_evt_path = miss.get_evt_list_path(obs_id)
                print(og_evt_path)
                copyfile(og_evt_path, new_evt_path)
                cur_evt_success[obs_id] = {i: True for i in miss.chosen_instruments if i in rel_act_insts}

            elif not miss.one_inst_per_obs:
                for inst in miss.chosen_instruments:
                    # TODO Change the se entry when possible
                    new_name = "obsid{oi}_inst{i}_subexp{se}.fits".format(oi=obs_id, i=inst, se=None)
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
                new_name = "obsid{oi}_inst{i}_subexp{se}.fits".format(oi=obs_id, i=inst, se=None)
                new_evt_path = arch.construct_processed_data_path(miss, obs_id) + new_name
                copyfile(og_evt_path, new_evt_path)
                cur_evt_success[obs_id][inst] = True


