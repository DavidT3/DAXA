import unittest

from astropy.coordinates import FK5
from astropy.units import Quantity

from daxa.mission import INTEGRALPointed

defaults = INTEGRALPointed()

class TestIntegralPointed(unittest.TestCase):
    def test_valid_inst_selection(self):
        mission = INTEGRALPointed(insts=['JEMX1', 'JEMX2'])
        self.assertEqual(mission.chosen_instruments, ['JEMX1', 'JEMX2'])
    
    def test_wrong_insts(self):
        with self.assertRaises(ValueError):
            INTEGRALPointed(insts=['wrong'])

    def test_name(self):
        self.assertEqual(defaults.name, 'integral_pointed')
    
    def test_coord_frame(self):
        self.assertEqual(defaults.coord_frame, FK5)

    def test_id_regex(self):
        self.assertEqual(defaults.id_regex, '^[0-9]{12}$')
    
    def test_fov(self):
        self.assertEqual(defaults.fov['JEMX1'], Quantity(2.4, 'deg'))
    


if __name__ == '__main__':
    unittest.main()
