import unittest

from astropy.coordinates import ICRS
from astropy.units import Quantity

from daxa.mission import Chandra

defaults = Chandra()

class TestChandra(unittest.TestCase):
    def test_valid_inst_selection(self):
        mission = Chandra(insts=['ACIS-I', 'ACIS-S'])
        self.assertEqual(mission.chosen_instruments, ['ACIS-I', 'ACIS-S'])

    def test_not_implemented_insts(self):
        with self.assertRaises(NotImplementedError):
            Chandra(insts='HETG')
        with self.assertRaises(NotImplementedError):
            Chandra(insts='LETG')
        
    def test_wrong_insts(self):
        with self.assertRaises(ValueError):
            Chandra(insts=['wrong'])

    def test_name(self):
        self.assertEqual(defaults.name, 'chandra')
    
    def test_coord_frame(self):
        self.assertEqual(defaults.coord_frame, ICRS)
    
    def test_fov(self):
        with self.assertWarns(UserWarning):
            self.assertEqual(defaults.fov['ACIS-I'], Quantity(27.8, 'arcmin'))


if __name__ == '__main__':
    unittest.main()
