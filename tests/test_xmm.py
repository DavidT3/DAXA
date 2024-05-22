import unittest

from astropy.coordinates import FK5
from astropy.units import Quantity

from daxa.mission import XMMPointed

class TestXMMPointed(unittest.TestCase):
    def setUp(self):
        self.defaults = XMMPointed()

    def test_valid_inst_selection(self):
        # Checking that inst argument is working correctly
        mission = XMMPointed(insts=['M1', 'M2'])
        self.assertEqual(mission.chosen_instruments, ['M1', 'M2'])
    
    def test_valid_inst_selection_alt_names(self):
        # Alternative instrument names should be able to be parsed
        with self.assertWarns(UserWarning):
            mission = XMMPointed(insts=['MOS1', 'MOS2'])

        self.assertEqual(mission.chosen_instruments, ['M1', 'M2'])
    
    def test_wrong_insts(self):
        # Shouldnt be able to declare an invalid instrument
        with self.assertRaises(ValueError):
            XMMPointed(insts=['wrong'])

    # the basic properties of the class are returning what is expected
    def test_name(self):
        self.assertEqual(self.defaults.name, 'xmm_pointed')
    
    def test_coord_frame(self):
        self.assertEqual(self.defaults.coord_frame, FK5)

    def test_id_regex(self):
        self.assertEqual(self.defaults.id_regex, '^[0-9]{10}$')
    
    def test_fov(self):
        self.assertEqual(self.defaults.fov, Quantity(15, 'arcmin'))
    


if __name__ == '__main__':
    unittest.main()
