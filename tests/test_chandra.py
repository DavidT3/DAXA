import unittest

from astropy.coordinates import ICRS
from astropy.units import Quantity

from daxa.mission import Chandra

# Would usually put this in a setUp() function, but it takes some time to instantiate
# Putting the mission object up here instead saves time when running the tests
defaults = Chandra()

class TestChandra(unittest.TestCase):
    def test_valid_inst_selection(self):
        # Checking that inst argument is working correctly
        mission = Chandra(insts=['ACIS-I', 'ACIS-S'])
        self.assertEqual(mission.chosen_instruments, ['ACIS-I', 'ACIS-S'])
        
    def test_wrong_insts(self):
        # Shouldnt be able to declare an invalid instrument
        with self.assertRaises(ValueError):
            Chandra(insts=['wrong'])

    # the basic properties of the class are returning what is expected
    def test_name(self):
        self.assertEqual(defaults.name, 'chandra')
    
    def test_coord_frame(self):
        self.assertEqual(defaults.coord_frame, ICRS)
    
    def test_fov(self):
        with self.assertWarns(UserWarning):
            self.assertEqual(defaults.fov['ACIS-I'], Quantity(27.8, 'arcmin'))


if __name__ == '__main__':
    unittest.main()
