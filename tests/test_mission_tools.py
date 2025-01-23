import unittest

from astropy.coordinates import SkyCoord
from astropy.units import Quantity

from daxa.mission.tools import multi_mission_filter_on_positions
from daxa.mission import *


class TestMultiMissionFilterOnPositions(unittest.TestCase):
    def setUp(self):
        self.pos = SkyCoord([232.474, 208.777], [66.075, 54.304], unit='deg')

    def test_dodgy_mission_in_search_dist(self):
        with self.assertRaises(ValueError):
            multi_mission_filter_on_positions(self.pos, {'wrong_mission': Quantity(30, 'arcmin')})

    def test_dodgy_search_dist_type(self):
        with self.assertRaises(ValueError):
            multi_mission_filter_on_positions(self.pos, ['wrong_type'])

    def test_search_dist_quantity_input(self):
        multi_mission_filter_on_positions(self.pos, Quantity(30, 'arcmin'))

    def test_search_dist_default_input(self):
        multi_mission_filter_on_positions(self.pos)

    def test_dodgy_mission_input(self):
        with self.assertRaises(ValueError):
            multi_mission_filter_on_positions(self.pos, missions=2)

    def test_dodgy_mission_input_within_list(self):
        with self.assertRaises(ValueError):
            multi_mission_filter_on_positions(self.pos, missions=['erosita_all_sky_de_dr1', 'wrong',
                                                              'rosat_pointed'])
        
    def test_some_missions_given(self):
        multi_mission_filter_on_positions(self.pos, missions=['erosita_all_sky_de_dr1', 
                                                              'rosat_pointed',
                                                              'swift',
                                                               'rosat_all_sky'])
    def test_some_search_dists_given(self):
        multi_mission_filter_on_positions(self.pos, 
                                          search_distance={'swift': Quantity(30, 'arcmin'),
                                          'rosat_pointed':  Quantity(30, 'arcmin')})
