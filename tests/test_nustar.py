import unittest

from astropy.coordinates import FK5
from astropy.units import Quantity

from daxa.mission import NuSTARPointed

defaults = NuSTARPointed()

class TestNuSTARPointed(unittest.TestCase):
    def test_valid_inst_selection(self):
        mission = NuSTARPointed(insts=['FPMA'])
        self.assertEqual(mission.chosen_instruments, ['FPMA'])
    
    def test_valid_inst_selection_alt_names(self):
        with self.assertWarns(UserWarning):
            mission = NuSTARPointed(insts=['FA', 'FB'])

        self.assertEqual(mission.chosen_instruments, ['FPMA', 'FPMB'])
    
    def test_wrong_insts(self):
        with self.assertRaises(ValueError):
            NuSTARPointed(insts=['wrong'])

    def test_name(self):
        self.assertEqual(defaults.name, 'nustar_pointed')
    
    def test_coord_frame(self):
        self.assertEqual(defaults.coord_frame, FK5)

    def test_id_regex(self):
        self.assertEqual(defaults.id_regex, '^[0-9]{11}$')
    
    def test_fov(self):
        self.assertEqual(defaults.fov, Quantity(6.5, 'arcmin'))
    

if __name__ == '__main__':
    unittest.main()