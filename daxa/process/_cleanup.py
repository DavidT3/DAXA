#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 04/09/2024, 13:10. Copyright (c) The Contributors
import os
import shutil
from functools import wraps
from typing import Union, List

import numpy as np

from daxa.archive import Archive


def _last_process(mission_names: Union[str, List[str]], obs_ident_num_comp: int):
    """
    This is a wrapper that should be applied to the final processing function for a mission. Note that other
    functions can be applied after that generate things, but the wrapped function should be the last thing that
    actually makes changes to the base data. This wrapper will assess which observations were complete failures and
    hold no useful data, noting that information in the archive and moving the data away from the
    successful observations

    :param str/List[str] mission_names: The allowed mission names that can possibly be processed by the function
        that _last_process is being used to wrap. The missions in this argument do not necessarily have to be
        present in the Archive.
    :param int obs_ident_num_comp: The number of identifier components needed to retrieve information produced by the
        function wrapped by this _last_process function. So for instance, when wrapping the last XMM processing step
        this would be set to 2, because final event lists are generated on an instrument level, so you need an ObsID
        and an instrument name to access them.
    """
    def last_process_function(proc_func):
        @wraps(proc_func)
        def wrapper(*args, **kwargs):

            # This runs the actual processing step that this decorator is wrapped around - obviously that needs
            #  to happen
            proc_func(*args, **kwargs)

            # We check if the first argument to the initial processing function was an archive. If not then either
            #  the processing function has been designed incompatibly, or this has been used to decorate the
            #  wrong thing
            if not isinstance(args[0], Archive):
                raise TypeError("The first argument of the processing function should be an archive.")

            # TODO REMOVE
            arch: Archive = args[0]

            # This will be used to store the final ObsID-level boolean flags that eventually get sent to the
            #  final_process_success of an Archive, telling it which ObsIDs ultimately have any useful data in them
            final_judgement = {}
            # Cycles through the possible mission names that the last processing function wrapped by _last_process
            #  could be applied to
            for name in mission_names:
                # If said name was in the archive then we go further
                if name in arch.mission_names:
                    # Set that mission to fully processed
                    arch[name].processed = True

                    # We get all the observations to could possibly have been processed (this won't include ones
                    #  which were excluded as entirely invalid, for instance where no valid instruments were active
                    #  or filters were in CalClosed position
                    # Then use the obs_ident_num_comp parameter (explained at the top of the function) to grab the
                    #  number of identifier components necessary to access success information on the output of the
                    #  wrapped function
                    obs = np.array(arch.get_obs_to_process(name))[:, :obs_ident_num_comp]
                    # The identifiers stored in obs are now passed to the archive method which can check which of
                    #  them had successful runs of the processing function which this function wraps
                    success_arr = arch.check_dependence_success(name, obs, proc_func.__name__)
                    # That knowledge of which had successful runs of the wrapped processing function is then used to
                    #  create a list of successful obs
                    success_obs = obs[success_arr, 0]
                    all_oid = arch[name].filtered_obs_ids.copy()
                    # From there we can identify the observation directories we should remove from the archive data
                    #  structure
                    oid_to_remove = all_oid[~np.isin(all_oid, success_obs)]
                    # Store a dictionary of which ObsIDs have any useful data after the final processing step
                    final_judgement[name] = {o_id: o_id not in oid_to_remove for o_id in all_oid}

            # Add the judgement dictionary to the appropriate archive property setter - we do this before moving
            #  stuff because some of the information in the final_process_success property is used in the
            #  construction of paths to failed observation
            arch.final_process_success = final_judgement

            # Iterating through the success dictionary - top levels are mission names. This is all quite cyclical and
            #  I probably could have set this up in a more elegant way but oh well
            for mn in final_judgement:
                for obs_id in arch.final_process_success[mn]:
                    if not arch.final_process_success[mn][obs_id]:
                        # If the observation was a total failure, we move its directory away from all the successful
                        #  data, to make the structure of our archive a bit nicer. We can use archive methods
                        #  to get the appropriate paths
                        cur_path = arch.construct_processed_data_path(mn, obs_id)
                        new_path = arch.construct_failed_data_path(mn, obs_id)

                        # We check if it exists because this run through could be from an archive that has been updated
                        #  and we won't be able to move any previously-run but failed ObsID dirs because they will
                        #  already have been moved
                        if not os.path.exists(new_path):
                            # Then can use shutil to move the failed ObsID and whatever might be in the directory
                            shutil.move(cur_path, new_path)

            # We automatically save after these checks - though there will have been a save at the end of the run of
            #  whatever process triggered this final check
            arch.save()

        return wrapper
    return last_process_function
