import unittest

from astropy.coordinates import FK5
from astropy.units import Quantity

from daxa.mission import Swift

defaults = Swift()

class TestSwift(unittest.TestCase):
    def test_valid_inst_selection(self):
        mission = Swift(insts=['XRT'])
        self.assertEqual(mission.chosen_instruments, ['XRT'])
    
    def test_wrong_insts(self):
        with self.assertRaises(ValueError):
            Swift(insts=['wrong'])

    def test_name(self):
        self.assertEqual(defaults.name, 'swift')
    
    def test_coord_frame(self):
        self.assertEqual(defaults.coord_frame, FK5)

    def test_id_regex(self):
        self.assertEqual(defaults.id_regex, '^[0-9]{11}$')
    
    def test_fov(self):
        self.assertEqual(defaults.fov['XRT'], Quantity(11.8, 'arcmin'))
    

if __name__ == '__main__':
    unittest.main()
