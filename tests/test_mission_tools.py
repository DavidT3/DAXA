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
    
    def test_insts_dodgy_input(self):
        """
        Invalid mission names should fail.
        """
        with self.assertRaises(ValueError):
            multi_mission_filter_on_positions(self.pos, insts={'dodgy': 'inst'})

    def test_insts_input_incl_mission_not_selected(self):
        """
        If user selects instruments for missions that arent in the missions argument, it should
        error.
        """
        with self.assertRaises(ValueError):
            multi_mission_filter_on_positions(self.pos, missions=['erosita_all_sky_de_dr1',
                                                                  'rosat_pointed'],
                                                           insts={'swift': 'inst'})

    def test_some_missions_have_insts_inputs(self):
        """
        Making sure the insts argument works even when some missions have instruments selected
        and others dont.
        """
        res =  multi_mission_filter_on_positions(self.pos, missions=['rosat_pointed',
                                                                     'swift'],
                                                           insts={'swift': 'XRT'})
        
        chosen_insts = [inst for miss in res for inst in miss.chosen_instruments]

        assert 'XRT' in chosen_insts
        assert 'BAT' not in chosen_insts
                                        
if __name__ == '__main__':
    unittest.main()

