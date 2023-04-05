#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 04/04/2023, 22:13. Copyright (c) The Contributors

from functools import wraps
from typing import Union, List

import numpy as np

from daxa.archive import Archive


def _last_process(mission_names: Union[str, List[str]]):
    """

    :param str/List[str] mission_names: The allowed mission names that can possibly be processed by the function
        that _last_process is being used to wrap. The missions in this argument do not necessarily have to be
        present in the Archive.
    :return:
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

            #
            for name in mission_names:
                if name in arch.mission_names:
                    arch[name].processed = True

                obs = np.unique(np.array(arch.get_obs_to_process(name))[:, 0])
                print(obs)
                print(arch.check_dependence_success(name, obs, proc_func.__name__))
                print('\n\n\n')

        return wrapper
    return last_process_function
