import unittest

from astropy.coordinates import FK5
from astropy.units import Quantity

from daxa.mission import ROSATAllSky, ROSATPointed

# Would usually put this in a setUp() function, but it takes some time to instantiate
# Putting the mission object up here instead saves time when running the tests
pnt_defaults = ROSATPointed()
sky_defaults = ROSATAllSky()

class TestROSATPointed(unittest.TestCase):
    def test_valid_inst_selection(self):
        # Checking that inst argument is working correctly
        mission = ROSATPointed(insts=['PSPCB', 'PSPCC'])
        self.assertEqual(mission.chosen_instruments, ['PSPCB', 'PSPCC'])
    
    def test_valid_inst_selection_alt_names(self):
        # Alternative instrument names should be able to be parsed
        with self.assertWarns(UserWarning):
            mission = ROSATPointed(insts=['PSPC-B', 'PSPC-C'])

        self.assertEqual(mission.chosen_instruments, ['PSPCB', 'PSPCC'])
    
    def test_wrong_insts(self):
        # Shouldnt be able to declare an invalid instrument
        with self.assertRaises(ValueError):
            ROSATPointed(insts=['wrong'])

    # the basic properties of the class are returning what is expected
    def test_name(self):
        self.assertEqual(pnt_defaults.name, 'rosat_pointed')
    
    def test_coord_frame(self):
        self.assertEqual(pnt_defaults.coord_frame, FK5)

    def test_id_regex(self):
        self.assertEqual(pnt_defaults.id_regex, r'^(RH|rh|RP|rp|RF|rf|WH|wh|WP|wp|WF|wf)\d{6}([A-Z]\d{2}|)$')
    
    def test_fov(self):
        self.assertEqual(pnt_defaults.fov['PSPCB'], Quantity(60, 'arcmin'))
    
class TestROSATALLSky(unittest.TestCase):
    def test_wrong_insts(self):
        # Shouldnt be able to declare an invalid instrument
        with self.assertRaises(ValueError):
            ROSATPointed(insts=['wrong'])

    # the basic properties of the class are returning what is expected
    def test_name(self):
        self.assertEqual(sky_defaults.name, 'rosat_all_sky')
    
    def test_coord_frame(self):
        self.assertEqual(sky_defaults.coord_frame, FK5)

    def test_id_regex(self):
        self.assertEqual(sky_defaults.id_regex, r'^(RS|rs)\d{6}[A-Z]\d{2}$')
    
    def test_fov(self):
        self.assertEqual(sky_defaults.fov, Quantity(180, 'arcmin'))


if __name__ == '__main__':
    unittest.main()
