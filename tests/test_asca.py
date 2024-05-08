import unittest

from astropy.coordinates import FK5
from astropy.units import Quantity

from daxa.mission import ASCA

# Would usually put this in a setUp() function, but it takes some time to instantiate
# Putting the mission object up here instead saves time when running the tests
defaults = ASCA()

class TestASCA(unittest.TestCase):
    def test_valid_inst_selection(self):
        # Checking that inst argument is working correctly
        mission = ASCA(insts=['SIS0', 'SIS1'])
        self.assertEqual(mission.chosen_instruments, ['SIS0', 'SIS1'])
    
    def test_valid_inst_selection_alt_names(self):
        # Alternative instrument names should be able to be parsed
        with self.assertWarns(UserWarning):
            mission = ASCA(insts=['S0', 'S1'])
        self.assertEqual(mission.chosen_instruments, ['SIS0', 'SIS1'])
    
    def test_wrong_insts(self):
        # Shouldnt be able to declare an invalid instrument
        with self.assertRaises(ValueError):
            ASCA(insts=['wrong'])

    # the basic properties of the class are returning what is expected
    def test_name(self):
        self.assertEqual(defaults.name, 'asca')
    
    def test_coord_frame(self):
        self.assertEqual(defaults.coord_frame, FK5)

    def test_id_regex(self):
        self.assertEqual(defaults.id_regex, '^[0-9]{8}$')
    
    def test_fov(self):
        self.assertEqual(defaults.fov['SIS0'], Quantity(11, 'arcmin'))


if __name__ == '__main__':
    unittest.main()
