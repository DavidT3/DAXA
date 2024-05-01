import unittest

from astropy.coordinates import FK5
from astropy.units import Quantity

from daxa.mission import ASCA

defaults = ASCA()

class TestASCA(unittest.TestCase):
    def test_valid_inst_selection(self):
        mission = ASCA(insts=['SIS0', 'SIS1'])
        self.assertEqual(mission.chosen_instruments, ['SIS0', 'SIS1'])
    
    def test_valid_inst_selection_alt_names(self):
        with self.assertWarns(UserWarning):
            mission = ASCA(insts=['S0', 'S1'])
        self.assertEqual(mission.chosen_instruments, ['SIS0', 'SIS1'])
    
    def test_wrong_insts(self):
        with self.assertRaises(ValueError):
            ASCA(insts=['wrong'])

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
