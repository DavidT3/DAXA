#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 23/11/2022, 15:23. Copyright (c) The Contributors


class DAXAConfigError(Exception):
    def __init__(self, *args):
        """
        Exception raised for flawed DAXA config files.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{0} '.format(self.message)
        else:
            return 'DAXAConfig has been raised'
