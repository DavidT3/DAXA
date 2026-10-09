#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (djturner@umbc.edu) 10/9/26, 9:52 AM. Copyright (c) The Contributors.

import unittest
from daxa.mission import MISS_INDEX


class TestMissionInstanceDeclaration(unittest.TestCase):
    """
    Simple test of whether each mission can be declared successfully - meant to be run regularly to make
    sure that there haven't been changes on the remote data sources we make use of in DAXA.
    """

    def check_missions_declares(self, name) -> None:
        """
        Constructor for the simplest possible test of every mission class - can it be declared? This method
        isn't run directly, but one per mission is added to this TestCase subclass. This means we don't
        have to remember to add a new test method for a new mission.
        """
        # Pull out the relevant mission class
        cur_miss_class = MISS_INDEX[name]

        # Declare the instance
        cur_miss = cur_miss_class()

        # Very simple check that there are actually some observations associated
        self.assertGreater(len(cur_miss), 1, f"{cur_miss.pretty_name} does not have any observations associated.")


# Dynamically attach mission declaration tests for every mission
#  Avoids us having to add a new method for every new mission class we add to DAXA.
for mission_name in MISS_INDEX:
    # All the tests that check that an EventList can be declared and an image can be generated from them
    gen_method = f"test_declare_{mission_name}"

    def create_declare_test(m_name):
        return lambda self: self.check_missions_declares(m_name)

    setattr(TestMissionInstanceDeclaration, gen_method, create_declare_test(mission_name))