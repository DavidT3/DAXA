import unittest

from astropy.coordinates import FK5
from astropy.units import Quantity

from daxa.mission import Suzaku

defaults = Suzaku()

class TestSuzaku(unittest.TestCase):
    def test_valid_inst_selection(self):
        mission = Suzaku(insts=['XIS0', 'XIS1'])
        self.assertEqual(mission.chosen_instruments, ['XIS0', 'XIS1'])
    
    def test_valid_inst_selection_alt_names(self):
        with self.assertWarns(UserWarning):
            mission = Suzaku(insts=['X0', 'X1'])

        self.assertEqual(mission.chosen_instruments, ['XIS0', 'XIS1'])

    def test_wrong_insts(self):
        with self.assertRaises(ValueError):
            Suzaku(insts=['wrong'])

    def test_name(self):
        self.assertEqual(defaults.name, 'suzaku')
    
    def test_coord_frame(self):
        self.assertEqual(defaults.coord_frame, FK5)

    def test_id_regex(self):
        self.assertEqual(defaults.id_regex, '^[0-9]{9}$')
    
    def test_fov(self):
        self.assertEqual(defaults.fov, Quantity(9.5, 'arcmin'))


if __name__ == '__main__':
    unittest.main()
