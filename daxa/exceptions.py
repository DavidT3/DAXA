#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 02/09/2024, 16:59. Copyright (c) The Contributors


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


class DAXADownloadError(Exception):
    def __init__(self, *args):
        """
        Exception raised for problems with data downloads orchestrated by DAXA.

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
            return 'DAXADownloadError has been raised'


class DAXANotDownloadedError(Exception):
    def __init__(self, *args):
        """
        Exception raised for when something attempts to perform an action that requires data to be downlaoded, and
        it hasn't been.

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
            return 'DAXANotDownloadedError has been raised'


class DuplicateMissionError(Exception):
    def __init__(self, *args):
        """
        Exception raised when multiple instances of the same mission are passed to an Archive definition.

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
            return 'DuplicateMissionError has been raised'


class ArchiveExistsError(Exception):
    def __init__(self, *args):
        """
        Exception raised when an archive name that has already been used in a particular
        DAXA output directory is used again.

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
            return 'ArchiveExistsError has been raised'


class MissionLockedError(Exception):
    def __init__(self, *args):
        """
        Exception raised when a mission instance has been locked (no further changes to selected observations
        can be made) and a change of some kind is attempted.

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
            return 'MissionLockedError has been raised'


class SASNotFoundError(Exception):
    def __init__(self, *args):
        """
        Exception raised if the XMM Scientific Analysis System can not be found on the system.

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
            return 'SASNotFoundError has been raised'


class SASVersionError(Exception):
    def __init__(self, *args):
        """
        Exception raised if the XMM Scientific Analysis System located on the system is a version
        that is not compatible with DAXA.

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
            return 'SASVersionError has been raised'

class eSASSNotFoundError(Exception):
    def __init__(self, *args):
        """
        Exception raised if the eROSITA Science Analysis Software System can not be found on the system.

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
            return 'eSASSNotFoundError has been raised'


class BackendSoftwareError(Exception):
    def __init__(self, *args):
        """
        Exception raised if a required piece of backend software has not been located.

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
            return 'BackendSoftwareError has been raised'


class NoXMMMissionsError(Exception):
    def __init__(self, *args):
        """
        Exception raised if an archive containing no XMM missions is passed to an XMM specific processing function.

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
            return 'NoXMMMissionsError has been raised'

class NoEROSITAMissionsError(Exception):
    def __init__(self, *args):
        """
        Exception raised if an archive containing no eROSITA missions is passed to an eROSITA specific processing function.

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
            return 'NoEROSITAMissionsError has been raised'


class NoProcessingError(Exception):
    def __init__(self, *args):
        """
        Exception raised if a method tries to access processed data when no processing has been applied.

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
            return 'NoProcessingError has been raised'


class NoDependencyProcessError(Exception):
    def __init__(self, *args):
        """
        Exception raised if a processing method that the current process depends on has not been run.

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
            return 'NoDependencyProcessError has been raised'


class NoObsAfterFilterError(Exception):
    def __init__(self, *args):
        """
        Exception raised if a there are no valid observations left in a mission after filtering processes have been
        applied.

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
            return 'NoObsAfterFilterError has been raised'


class IllegalSourceType(Exception):
    def __init__(self, *args):
        """
        Exception raised if a source type that isn't in the DAXA source type taxonomy has been used.

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
            return 'IllegalSourceType has been raised'


class NoTargetSourceTypeInfo(Exception):
    def __init__(self, *args):
        """
        Exception raised if a mission doesn't have any information on each observation's target source type.

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
            return 'NoSourceTypeInfo has been raised'


class ObsNotAssociatedError(Exception):
    def __init__(self, *args):
        """
        Exception raised if an observation is not associated with a particular mission's filtered dataset.

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
            return 'ObsNotAssociatedError has been raised'


class MissionNotAssociatedError(Exception):
    def __init__(self, *args):
        """
        Exception raised if a mission is not associated with a particular archive dataset.

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
            return 'MissionNotAssociatedError has been raised'


class NoRegionsAssociatedError(Exception):
    def __init__(self, *args):
        """
        Exception raised if there are no source regions available for the user to retrieve.

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
            return 'NoRegionsAssociatedError has been raised'


class IncompatibleSaveError(Exception):
    def __init__(self, *args):
        """
        Exception raised if a save file being read in to reinstate a DAXA mission or archive is being used incorrectly
        and is not compatible with the process.

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
            return 'IncompatibleSaveError has been raised'


class PreProcessedNotSupportedError(Exception):
    def __init__(self, *args):
        """
        Exception raised if the user attempts to access pre-processed data for a mission class that does not support
        it (usually because the data are not available in the archive).

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
            return 'PreProcessedNotSupportedError has been raised'


class PreProcessedNotAvailableError(Exception):
    def __init__(self, *args):
        """
        Exception raised if the user attempts to access a pre-processed product that is not available for the
        mission - e.g. if they try to access an image for an energy band not present in the archive.

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
            return 'PreProcessedNotAvailableError has been raised'


class DAXADeveloperError(Exception):
    def __init__(self, *args):
        """
        Raised when an error has occurred that needs the attention of developers.

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
            return 'DAXADeveloperError has been raised'
