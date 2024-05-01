import unittest

from astropy.coordinates import FK5
from astropy.units import Quantity

from daxa.mission import XMMPointed

class TestXMMPointed(unittest.TestCase):
    def setUp(self):
        self.defaults = XMMPointed()

    def test_valid_inst_selection(self):
        mission = XMMPointed(insts=['M1', 'M2'])
        self.assertEqual(mission.chosen_instruments, ['M1', 'M2'])
    
    def test_valid_inst_selection_alt_names(self):
        with self.assertWarns(UserWarning):
            mission = XMMPointed(insts=['MOS1', 'MOS2'])

        self.assertEqual(mission.chosen_instruments, ['M1', 'M2'])

    def test_OM_error(self):
        with self.assertRaises(NotImplementedError):
            XMMPointed(insts='OM')
    
    def test_wrong_insts(self):
        with self.assertRaises(ValueError):
            XMMPointed(insts=['wrong'])

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
