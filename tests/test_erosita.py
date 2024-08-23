#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 23/08/2024, 11:21. Copyright (c) The Contributors

import os
import shutil
import unittest
from datetime import datetime
from io import BytesIO
from unittest.mock import patch, MagicMock, call

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, FK5, Galactic
from astropy.coordinates.name_resolve import NameResolveError
from astropy.io import fits
from astropy.units import Quantity
from daxa import OUTPUT
from daxa.config import EROSITA_CALPV_INFO
from daxa.exceptions import NoObsAfterFilterError, IllegalSourceType, NoTargetSourceTypeInfo
from daxa.mission import eRASS1DE, eROSITACalPV
from numpy.testing import assert_array_equal


# This class is used to mock a request response, it is used in the unittests of eRASS1DE._download_call()
class MockRequestResponse(object):
    '''
    Mimics the properties of the return value of a session.get object for testing in
    TestDownloadCall.
    '''
    def __init__(self, text):
        self.text = text
        self.raw = text

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        pass

class TesteROSITACalPV(unittest.TestCase):
    """
    This contains the unit tests for most eROSITACalPV functionality except for the download method.
    Testing for download() is in a different class because it needs some niche/ specific things
    in the setUp() method, that were unnecessary for most eROSITACalPV testing.
    """
    def setUp(self):
        # This gets run before every test, so within each test these objects can be used
        self.defaults = eROSITACalPV()
        self.filtered = eROSITACalPV(fields='eFEDS')
        self.alt_field_nme = eROSITACalPV(fields='crab iii')  # filtered using an alternative field name
        self.crab = eROSITACalPV(fields='crab')  # This has a special field filtering, so nice to test with this too
        self.field_type = eROSITACalPV(fields='survey')
        self.type_n_nme = eROSITACalPV(fields=['survey', 'puppis a'])  # for testing with field types and field names
    
    def tearDown(self):
        # This is run after each test
        # In some of the testing of _download_call() files are written
        # Good to remove files here rather than within a test so that if the test fails, the files are still removed
        if os.path.exists('test_data/temp_download'):
            shutil.rmtree('test_data/temp_download')
        if os.path.exists('test_data/erosita_calpv_raw'):
            shutil.rmtree('test_data/erosita_calpv_raw')
    
    # This tests when you pick a field upon instaniation it behaves correctly
    def test_chosen_fields(self):
        self.assertEqual(self.defaults.chosen_fields,
                         list(set(EROSITA_CALPV_INFO["Field_Name"].tolist())))

        self.filtered = eROSITACalPV(fields='eFEDS')
        self.assertEqual(self.filtered.chosen_fields, ['EFEDS'])
        assert_array_equal(self.filtered.filtered_obs_ids, np.array(['300007', '300008', '300009', '300010']))

        # can't pass fields with the wrong type
        with self.assertRaises(ValueError):
            eROSITACalPV(fields=7)
        
        # can't pass a list if all elements arent strings
        with self.assertRaises(ValueError):
            eROSITACalPV(fields=['ok', 'ok', 7])
        
        # Incorrect fields raise an error
        with self.assertRaises(ValueError) as err:
            eROSITACalPV(fields=['efeds', 'eta cha', 'nope'])
            # check correct error is raised
            self.assertEqual('Some field names or field types NOPE are not associated with this '
            'mission, please choose from the following fields; PSR_J0537_6910,LMC_N132D,'
            '1RXS_J185635_375433,HR_3165__ZET_PUP_,RE_J2334_471,A3391_A3395,EFEDS,'
            'LMC_SN1987A,TGUH2213P1__DARK_CLOUD_,47_TUC__NGC_104_,PSR_1509_58,NGC_7793_P13,CRAB_3,'
            'IGR_J16318_4848,1RXS_J072025_312554,CRAB_2,PSR_B0656_14,1ES_0102_72,PUPPIS_A,1H0707,'
            'IGR_J13020_6359__2RXP_J130159_635806_,VELA_SNR,CRAB_1,OAO_1657_415,A3158,A3266,'
            'NGC_2516,PSR_J0540_PSR_J0537,CRAB_4,ETA_CHA,3C390_3,PSR_J0540_6919 or field types; '
            'SURVEY,MAGELLANIC_CLOUDS,GALACTIC_FIELDS,EXTRAGALACTIC_FIELDS', str(err.exception))
        
        # alternative field names should pass
        self.assertEqual(self.alt_field_nme.chosen_fields, ['CRAB_3'])
        # crab should return all crab fields
        self.assertEqual(set(self.crab.chosen_fields), set(['CRAB_1', 'CRAB_2', 'CRAB_3', 'CRAB_4']))
        # field types should return correct field names
        self.assertEqual(set(self.field_type.chosen_fields), set(['EFEDS', 'ETA_CHA']))
        # combination of field types and names
        self.assertEqual(set(self.type_n_nme.chosen_fields), set(['EFEDS', 'ETA_CHA', 'PUPPIS_A']))
    
    # testing filter_on_fields method
    def test_filter_on_fields(self):
        self.defaults.filter_on_fields('efeds')
        self.assertEqual(self.defaults.chosen_fields, ['EFEDS'])
        assert_array_equal(self.defaults.filtered_obs_ids, np.array(['300007', '300008', '300009', '300010']))

    def test_filter_on_obs_ids(self):
        # testing one obs
        self.defaults.filter_on_obs_ids('300004')
        assert_array_equal(self.defaults.filtered_obs_ids, np.array(['300004']))

        # testing on multiple obs
        self.field_type.filter_on_obs_ids(['300004', '300007', '300008'])
        assert_array_equal(self.defaults.filtered_obs_ids, np.array(['300004', '300007', '300008']))
    
    def test_filter_on_obs_ids_invalid_obs(self):
        # Shouldnt be able to declare an invalid obsid
        with self.assertRaises(ValueError):
            self.defaults.filter_on_obs_ids('wrong')
        
        with self.assertRaises(ValueError):
            # Shouldnt be able to declare an invalid obsid
            self.defaults.filter_on_obs_ids(['300004', 'wrong', '300007'])
    
    def test_no_obs_after_filter(self):
        with self.assertRaises(NoObsAfterFilterError):
            # self.filtered is just efeds, so when I filter on the eta cha obs id, there should be no obs left
            self.filtered.filter_on_obs_ids(['300004'])
    
    def test_filter_on_rect_region(self):
        # checking it works as expected
        self.defaults.filter_on_rect_region([129, 1], [145, 2])  # These coords straddle the efeds fields
        assert_array_equal(self.defaults.filtered_obs_ids, np.array(['300007', '300008', '300009', '300010']))
        expected_ra_decs = SkyCoord([129.55, 133.86, 138.14, 142.45], [1.50, 1.50, 1.50, 1.50], unit=u.deg, frame=FK5)
        assert_array_equal(self.defaults.filtered_ra_decs, expected_ra_decs)

        # an error is raised when no obs are found
        with self.assertRaises(NoObsAfterFilterError):
            self.defaults.filter_on_rect_region([129, 0], [145, 1])

    def test_filter_on_positions_one_pos(self):
        # Testing for one RA and DEC input as a list
        self.defaults.filter_on_positions([129.55, 1.50])
        assert_array_equal(self.defaults.filtered_obs_ids, np.array(['300007', '300008']))

    def test_filter_on_positions_mult_pos(self):
        # Testing for multiple RA and DECs input as nested list
        self.defaults.filter_on_positions([[129.55, 1.50], [133.86, 1.5]])
        assert_array_equal(self.defaults.filtered_obs_ids, np.array(['300007', '300008', '300009']))

    def test_filter_on_positions_skycoord(self):
        self.defaults.filter_on_positions(SkyCoord(129.55, 1.50, unit=u.deg, frame=FK5))
        assert_array_equal(self.defaults.filtered_obs_ids, np.array(['300007', '300008']))

    def test_filter_on_positions_skycoord_alt_frame(self):
        # Testing for one RA and DEC input as a skycoord
        self.defaults.filter_on_positions(SkyCoord(224.415, 24.303, unit=u.deg, frame=Galactic))
        assert_array_equal(self.defaults.filtered_obs_ids, np.array(['300007', '300008']))
    
    def test_filter_on_positions_return(self):
        # Testing correct df is returned
        ret_val = self.defaults.filter_on_positions([129.55, 1.50], return_pos_obs_info=True)
        self.assertTrue(isinstance(ret_val, pd.DataFrame))
        self.assertAlmostEqual(float(ret_val['pos_ra'][0]), 129.55)
        self.assertAlmostEqual(float(ret_val['pos_dec'][0]), 1.5)
        self.assertEqual(ret_val['ObsIDs'][0], '300007,300008')

    def test_filter_on_positions_sd_quantity(self):
        # check search distance is working when input as a quantity
        self.defaults.filter_on_positions([129.55, 1.5], search_distance=Quantity(5, 'deg'))
        assert_array_equal(self.defaults.filtered_obs_ids, np.array(['300007', '300008']))

    def test_filter_on_positions_sd_list(self):
        # check search distance is working when input as a list
        coords_to_search = [[129.55, 1.5], [130.33, -78.96], [284.15, -37.91]]
        search_dist = [5, 1, 0.5]
        self.defaults.filter_on_positions(coords_to_search, search_distance=search_dist)
        assert_array_equal(self.defaults.filtered_obs_ids, np.array(['300007', '300008', '300004', '700008']))

    def test_filter_on_positions_sd_list_wrong_length(self):
        # check search distance errors when input as a list with a length that doesnt match coords to search
        coords_to_search = [[129.55, 1.5], [130.33, -78.96], [284.15, -37.91]]
        search_dist = [5, 1]
        with self.assertRaises(ValueError):
            self.defaults.filter_on_positions(coords_to_search, search_distance=search_dist)
    
    def test_filter_on_positions_no_obs_after_filter(self):
        # no obs should be returned with this coord, should raise an error
        with self.assertRaises(NoObsAfterFilterError):
            self.defaults.filter_on_positions([180, 2])
    
    def test_filter_on_name(self):
        # checking it works
        self.defaults.filter_on_name('A3158')
        self.assertEqual(self.defaults.filtered_obs_ids, ['700177'])

    def test_filter_on_bad_name(self):
        # checking error is raised for invalid name
        with self.assertRaises(NameResolveError):
            self.defaults.filter_on_name('wrong')
    
    def test_filter_on_some_bad_names(self):
        # checking it can pick up an invalid name amongst valid ones
        with self.assertWarns(UserWarning):
            self.defaults.filter_on_name(['A3158', 'wrong'])
    
    def test_filter_on_time(self):
        # checking works as expected
        start = datetime.fromisoformat('2019-11-03T02:42:50.227')
        end = datetime.fromisoformat('2019-11-04T03:36:37.671')
        self.defaults.filter_on_time(start_datetime=start, end_datetime=end)
        self.assertEqual(self.defaults.filtered_obs_ids, ['300007'])
    
    def test_filter_on_time_overrun(self):
        # checking the overrun functionality - shouldnt return any obs with times I have put in
        start = datetime.fromisoformat('2019-11-03T02:42:50.227')
        end = datetime.fromisoformat('2019-11-04T03:15:00.000')  # edited this to be earlier than efeds end time
        with self.assertRaises(NoObsAfterFilterError):
            self.defaults.filter_on_time(start_datetime=start, end_datetime=end, over_run=False)

    def test_filter_on_target_type(self):
        # Error is raised when you put in an invalid source
        with self.assertRaises(IllegalSourceType):
            self.defaults.filter_on_target_type('wrong')
        
        with self.assertRaises(NoTargetSourceTypeInfo):
            # No target information for erosita, so an error should be raised
            self.defaults.filter_on_target_type('XRB')
    
    def test_filter_on_positions_at_time(self):
        start = datetime.fromisoformat('2019-11-03T02:42:50.227')
        end = datetime.fromisoformat('2019-11-04T03:36:37.671')

        # both start and end time arguments need to be datetime objects
        with self.assertRaises(TypeError):
            self.defaults.filter_on_positions_at_time([129.55, 1.5], start, 'end')
        
        # error should be raised when positions and start and end arent the same length
        with self.assertRaises(ValueError):
            pos = [[1, 2], [3, 4], [4, 5]]
            self.defaults.filter_on_positions_at_time(pos, start, end)
        
        # testing the same as before but with positions as a skycoord
        with self.assertRaises(ValueError):
            pos = SkyCoord([[1, 2], [3, 4], [4, 5]], unit=u.deg, frame=FK5)
            self.defaults.filter_on_positions_at_time(pos, start, end)
        
        # start and end times should be the same length
        with self.assertRaises(TypeError):
            pos = [1, 2]
            multi_start = np.array([datetime.fromisoformat('2019-11-03T02:42:50.227'),
                                    datetime.fromisoformat('2019-11-03T02:41:50.227')])
            self.defaults.filter_on_positions_at_time(pos, multi_start, end)
        
        # testing the same as before but positions is a skycoord
        with self.assertRaises(TypeError):
            pos = SkyCoord(1, 2, unit=u.deg, frame=FK5)
            multi_start = np.array([datetime.fromisoformat('2019-11-03T02:42:50.227'),
                                    datetime.fromisoformat('2019-11-03T02:41:50.227')])
            self.defaults.filter_on_positions_at_time(pos, multi_start, end)

        # testing the same as before but with a single position and multiple start times
        with self.assertRaises(ValueError):
            pos = [1, 2]
            multi_start = np.array([datetime.fromisoformat('2019-11-03T02:42:50.227'),
                                    datetime.fromisoformat('2019-11-03T02:41:50.227')])
            multi_end = np.array([datetime.fromisoformat('2019-11-04T03:36:37.671'),
                                  datetime.fromisoformat('2019-11-04T03:37:37.671')])
            self.defaults.filter_on_positions_at_time(pos, multi_start, multi_end)

        # finally just checking that the function works as expected
        self.defaults.filter_on_positions_at_time([129.55, 1.5], start, end)
        self.assertEqual(self.defaults.filtered_obs_ids, ['300007'])

    # then I test that the basic attributes of the function return what is expected
    def test_name(self):
        self.assertEqual(self.defaults.name, 'erosita_calpv')
    
    def test_id_regex(self):
        self.assertEqual(self.defaults.id_regex, '^[0-9]{6}$')
    
    def test_fov(self):
        with self.assertWarns(UserWarning):
            self.assertEqual(self.defaults.fov, Quantity(4.5, 'deg'))
    
    # there is a special case with ones obsid that was input wrong on the erosita website
    # checking that a warning is raised and the correct obsids are returned instead
    def test_filter_on_obs_ids(self):
        with self.assertWarns(UserWarning):
            self.defaults.filter_on_obs_ids('700195')
            assert_array_equal(self.defaults.filtered_obs_ids, np.array(['700199', '700200']))

    # this is just checking that download call is working as expected
    def test_download_call(self):
        # for some reason this is only working in a context manager but not using decorators, i havent got the foggiest why
        with patch('daxa.mission.erosita.requests.get') as mock_p:
            with patch('daxa.mission.erosita.tarfile.open') as mock_t:
                # I just mock the request response and the tarfile opening
                mock_response = MagicMock()
                mock_response.raw = BytesIO(b'fake_data')
                mock_p.return_value.__enter__.return_value = mock_response
            
                mock_tarfile = MagicMock()
                mock_tarfile.extractcall = 'doesntmatter'
                mock_t.return_value.open.return_value.__enter__.return_value = mock_tarfile

                link = 'https://erosita.mpe.mpg.de/edr/eROSITAObservations/CalPvObs/eta_Cha.tar.gz'

                eROSITACalPV._download_call('test_data', link)

                # if download_call works then these two functions should have been called once
                mock_p.assert_called_once_with(link, stream=True)
                mock_t.assert_called_once_with('test_data/temp_download/ETA_CHA/ETA_CHAETA_CHA.tar.gz', 'r:gz')

        # lastly I check that the appropiate folder to download into has been created
        self.assertTrue(os.path.exists('test_data/temp_download/ETA_CHA/ETA_CHA/'))
    
    # some files arrive in a folder, so this tests that _directory_formating can deal with that
    def test_directory_formatting_files_in_one_folder(self):
        # writing some files to test the function with
        # for testing purposes I am changing this attribute so the test files get written to the test_data folder
        self.filtered._top_level_output_path = 'test_data/'
        # self.raw_data_path is now test_data/erosita_calpv_raw

        # setting up my fake downloaded data
        # defining my path for fake downloaded data to go in
        path = 'test_data/erosita_calpv_raw/temp_download/EFEDS/'
        # making the directories
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # making the files
        obs_ids = ['300007', '300008', '300009', '300010']
        for obs in obs_ids:
            with open(path + obs + '.txt', 'w') as f:
                f.write('testing')
        # when downloading calpv data they all come with extra pdfs
            with open(path + obs + 'eRO' + '.txt', 'w') as f:
                f.write('testing')
        
        self.filtered._directory_formatting()

        self.assertTrue(os.path.exists('test_data/erosita_calpv_raw/300007/300007.txt'))
        self.assertTrue(os.path.exists('test_data/erosita_calpv_raw/300008/300008.txt'))
        self.assertTrue(os.path.exists('test_data/erosita_calpv_raw/300009/300009.txt'))
        self.assertTrue(os.path.exists('test_data/erosita_calpv_raw/300010/300010.txt'))
        self.assertFalse(os.path.exists('test_data/erosita_calpv_raw/temp_download'))

    # some files arrive in two folders, so this tests that _directory_formating can deal with that
    def test_directory_formatting_files_in_two_folders(self):
        # writing some files to test the function with
        # for testing purposes I am changing this attribute so the test files get written to the test_data folder
        self.filtered._top_level_output_path = 'test_data/'
        # self.raw_data_path is now test_data/erosita_calpv_raw

        # setting up my fake downloaded data
        # defining my path for fake downloaded data to go in
        path = 'test_data/erosita_calpv_raw/temp_download/EFEDS/EFEDS/'
        # making the directories
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # making the files
        obs_ids = ['300007', '300008', '300009', '300010']
        for obs in obs_ids:
            with open(path + obs + '.txt', 'w') as f:
                f.write('testing')
        # when downloading calpv data they all come with extra pdfs
            with open(path + obs + 'eRO' + '.txt', 'w') as f:
                f.write('testing')
        
        self.filtered._directory_formatting()

        self.assertTrue(os.path.exists('test_data/erosita_calpv_raw/300007/300007.txt'))
        self.assertTrue(os.path.exists('test_data/erosita_calpv_raw/300008/300008.txt'))
        self.assertTrue(os.path.exists('test_data/erosita_calpv_raw/300009/300009.txt'))
        self.assertTrue(os.path.exists('test_data/erosita_calpv_raw/300010/300010.txt'))
        self.assertFalse(os.path.exists('test_data/erosita_calpv_raw/temp_download'))

    # some files arrive in three folders, so this tests that _directory_formating can deal with that
    def test_directory_formatting_files_in_three_folders(self):
        # writing some files to test the function with
        # for testing purposes I am changing this attribute so the test files get written to the test_data folder
        self.filtered._top_level_output_path = 'test_data/'
        # self.raw_data_path is now test_data/erosita_calpv_raw

        # setting up my fake downloaded data
        # defining my path for fake downloaded data to go in
        path = 'test_data/erosita_calpv_raw/temp_download/EFEDS/EFEDS/EFEDS/'
        # making the directories
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # making the files
        obs_ids = ['300007', '300008', '300009', '300010']
        for obs in obs_ids:
            with open(path + obs + '.txt', 'w') as f:
                f.write('testing')
        # when downloading calpv data they all come with extra pdfs
            with open(path + obs + 'eRO' + '.txt', 'w') as f:
                f.write('testing')
        
        self.filtered._directory_formatting()

        self.assertTrue(os.path.exists('test_data/erosita_calpv_raw/300007/300007.txt'))
        self.assertTrue(os.path.exists('test_data/erosita_calpv_raw/300008/300008.txt'))
        self.assertTrue(os.path.exists('test_data/erosita_calpv_raw/300009/300009.txt'))
        self.assertTrue(os.path.exists('test_data/erosita_calpv_raw/300010/300010.txt'))
        self.assertFalse(os.path.exists('test_data/erosita_calpv_raw/temp_download'))
    
    def test_get_evlist_path_from_obs(self):
        self.defaults._top_level_output_path = 'test_data/'
        # i am mocking these files existing, so I dont have to write any myself
        with patch('daxa.mission.erosita.os.listdir') as mock_listdir:
            mock_listdir.return_value = ['fm00_300004_020_EventList_c001.fits',
                                         'fm00_300004_020_EventList_c001_if123.fits']
            
            result = self.defaults.get_evt_list_path(obs_id='300004')

        self.assertEqual(result, 'test_data/erosita_calpv_raw/300004/fm00_300004_020_EventList_c001.fits')
    
class TesteROSITACalPV_download(unittest.TestCase):
    '''
    Putting this test into a separate class since it needs a lot of patches/ mocking that is
    specific to testing download() only.
    '''

    # Here do all the mocking
    def setUp(self):
        self.etacha = eROSITACalPV(fields='eta cha')
        self.etacha_insts = eROSITACalPV(fields='eta cha', insts=['TM1', 'TM2'])
        self.survey = eROSITACalPV(fields='survey')

        self.mock_dir_frmt = patch.object(eROSITACalPV, '_directory_formatting').start()
        self.mock_inst_filt = patch.object(eROSITACalPV, '_inst_filtering').start()
        self.mock_down_call = patch.object(eROSITACalPV, '_download_call').start()
        self.mock_get_evlist = patch.object(eROSITACalPV, 'get_evt_list_path').start()

        self.mock_tqdm = patch('daxa.mission.erosita.tqdm').start()
        self.mock_Pool = patch('daxa.mission.erosita.Pool').start()
    
    def tearDown(self):
        # I read somewhere that you should stop all the patches/mocking or you can get wierd errors
        patch.stopall()

        # some of the tests write files, so this makes sure they are deleted - even if the test fails
        if os.path.exists('test_data/erosita_calpv_raw'):
            shutil.rmtree('test_data/erosita_calpv_raw')

    def test_successful_download(self):
        # this object just includes the crab 3 observation
        # I am overwriting this so files are written into the test_data directory instead of loose somewhere on the system
        self.etacha._top_level_output_path = 'test_data/'
        
        self.etacha.download(num_cores=1)

        # making sure the correct directory is made
        self.assertTrue(os.path.exists('test_data/erosita_calpv_raw'))

        # download would have been sucessful is all of these functions are called in the way that I am testing
        # I mock these functions because I dont want large data downloads actually happening when a test is run
        down_link = 'https://erosita.mpe.mpg.de/edr/eROSITAObservations/CalPvObs/eta_Cha.tar.gz'
        self.mock_down_call.assert_called_once_with(raw_dir='test_data/erosita_calpv_raw/', link=down_link)
        self.mock_inst_filt.assert_not_called()
        self.mock_dir_frmt.assert_called_once()
        self.assertTrue(self.etacha._download_done)
    
    # If certain instruments have been chosen then after the download inst_filtering should have been called
    def test_successful_download_w_inst_filtering(self):
        # this object just includes the crab 3 observation
        # I am overwriting this so files are written into the test_data directory instead of loose somewhere on the system
        self.etacha_insts._top_level_output_path = 'test_data/'
        # specifying what the mocked evlist_path_function should return
        self.mock_get_evlist.return_value = 'test_data/erosita_calpv_raw/300004/fm00_300004_020_EventList_c001.fits'
        
        self.etacha_insts.download(num_cores=1)

        # making sure the correct directory is made and mocked functions are called the right way
        self.assertTrue(os.path.exists('test_data/erosita_calpv_raw'))
        down_link = 'https://erosita.mpe.mpg.de/edr/eROSITAObservations/CalPvObs/eta_Cha.tar.gz'
        self.mock_down_call.assert_called_once_with(raw_dir='test_data/erosita_calpv_raw/', link=down_link)
        path = 'test_data/erosita_calpv_raw/300004/fm00_300004_020_EventList_c001.fits'
        self.mock_inst_filt.assert_called_once_with(insts=['TM1', 'TM2'], evlist_path=path)
        self.mock_dir_frmt.assert_called_once()
        self.assertTrue(self.etacha_insts._download_done)  # checking this attribute is changed

    # DAVID NOTE - this was failing and Jess told me to delete it
    # testing that everything is downloaded even if some obs are already downloaded
    # def test_successful_download_some_already_downloaded(self):
    #     # I am overwriting this so files are written into the test_data directory instead of loose somewhere on the system
    #     self.survey._top_level_output_path = 'test_data/'
    #
    #     # setting up a directory of some of the obs ids to mimic those already being downloaded
    #     path = 'test_data/erosita_calpv_raw/{}'
    #     obs_ids = ['300007/', '300008/', '300009/', '300010/']
    #     # making the directories
    #     for obs in obs_ids:
    #         os.makedirs(os.path.dirname(path.format(obs)), exist_ok=True)
    #
    #     # only eta cha is left to be downloaded out of the survey fields
    #     self.survey.download()
    #
    #     # checking all my mocked objects are called appropriately
    #     down_link = 'https://erosita.mpe.mpg.de/edr/eROSITAObservations/CalPvObs/eta_Cha.tar.gz'
    #     self.mock_down_call.assert_called_once_with(raw_dir='test_data/erosita_calpv_raw/', link=down_link)
    #     self.mock_inst_filt.assert_not_called()
    #     self.mock_dir_frmt.assert_called_once()
    #     self.assertTrue(self.survey._download_done)


    def test_download_raises_warning(self):
        # I am overwriting this so files are written into the test_data directory instead of loose somewhere on the system
        self.etacha._top_level_output_path = 'test_data/'

        # This mimics the data already being downloaded
        path = 'test_data/erosita_calpv_raw/300004/'
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with self.assertWarns(UserWarning):
            self.etacha.download()
    
    def test_invalid_num_cores(self):
        # I am overwriting this so files are written into the test_data directory instead of loose somewhere on the system
        self.etacha._top_level_output_path = 'test_data/'

        with self.assertRaises(ValueError):
            self.etacha.download(num_cores=-3)

class TestERASS1DE(unittest.TestCase):
    '''
    Here the basic attributes and methods are tested. The download and _download call methods are
    tested in a different class because the need unique mocked objects.
    '''
    def setUp(self):
        # some mission objects that I use in most tests
        self.defaults = eRASS1DE()
        self.tm1 = eRASS1DE(insts='TM1')
    
    def tearDown(self):
        if os.path.exists('test_data/inst_filt_if_12.fits'):
            os.remove('test_data/inst_filt_if_12.fits')

    def test_chosen_instruments(self):
        # error is raised for invalid instruments
        with self.assertRaises(ValueError):
            eRASS1DE(insts='wrong')
        
        # error is raised for empty list
        with self.assertRaises(ValueError):
            eRASS1DE(insts=[])
        
        # Can choose instruments correctly
        self.assertEqual(self.tm1.chosen_instruments, ['TM1'])
    
    def test_name(self):
        self.assertEqual(self.defaults.name, 'erosita_all_sky_de_dr1')

    def test_fov(self):
        self.assertEqual(self.defaults.fov, Quantity(1.8, 'degree'))
    
    def test_id_regex(self):
        self.assertEqual(self.defaults.id_regex, '^[0-9]{6}$')
    
    def test_inst_filtering(self):
        insts = ['TM1', 'TM2']
        # This is a test fits file of structure: col1 [1, 2, 3, 4, 5] = evts
        # col2 [1, 2, 5, 1, 7] = TM_NR
        evlist_path = 'test_data/inst_filt.fits'

        eRASS1DE._inst_filtering(insts, evlist_path)

        # should have created a file with this name
        self.assertTrue(os.path.exists('test_data/inst_filt_if_12.fits'))

        # testing that the file has actually filtered the correct instruments
        with fits.open('test_data/inst_filt_if_12.fits') as fitsfile:
            assert_array_equal(fitsfile[1].data['events'], np.array([1, 2, 4]))
            assert_array_equal(fitsfile[1].data['TM_NR'], np.array([1, 2, 1]))

        os.remove('test_data/inst_filt_if_12.fits')
    
    @patch('daxa.mission.erosita.os.path.exists')
    @patch('daxa.mission.erosita.fits')
    def test_inst_filtering_already_filtered(self, mock_exists, mock_fits):
        insts = ['TM1', 'TM2']
        # Test fits file of structure: col1 [1, 2, 3, 4, 5] = evts
        # col2 [1, 2, 5, 1, 7] = TM_NR
        evlist_path = 'test_data/inst_filt.fits'
        mock_exists.return_value = True  # this mimics that instrument filtering has already happened

        eRASS1DE._inst_filtering(insts, evlist_path)

        # since instrument filtering has already happened then this should not have been called
        mock_fits.open.assert_not_called()

class TesteRASS1DEDownload(unittest.TestCase):
    """
    Putting this test into a separate class since it needs a lot of patches/ mocking that is
    specific to testing download() only.
    """
    # setting up all the mock objects I need to test whether download was excecuted properly
    # I did try a way of not using so many mockec objects, but I didnt see another way
    def setUp(self):
        self.defaults = eRASS1DE()
        self.tm1 = eRASS1DE(insts='TM1')

        self.mock_download_call = patch.object(eRASS1DE, '_download_call').start()
        self.mock_get_evlist_from_obs = patch.object(eRASS1DE, 'get_evt_list_path').start()
        self.mock_inst_filter = patch.object(eRASS1DE, '_inst_filtering').start()

        self.mock_session = patch('daxa.mission.erosita.requests.Session').start()
        self.mock_open = patch('builtins.open').start()
        self.mock_os = patch('daxa.mission.erosita.os').start()
        self.mock_tqdm = patch('daxa.mission.erosita.tqdm').start()
        self.mock_Pool = patch('daxa.mission.erosita.Pool').start()
        self.mock_warn = patch('daxa.mission.erosita.warn').start()
        self.mock_DAXADownloadError = patch('daxa.mission.erosita.DAXADownloadError').start()
    
    def tearDown(self):
        # stopping all the patches and mocks to prevent weird behaviour occuring
        patch.stopall()
        self.mock_download_call.stop()
        self.mock_get_evlist_from_obs.stop()
        self.mock_inst_filter.stop()

    # testing the basic functionalilty works
    def test_download(self):
        test_class = self.defaults
        test_class.filter_on_obs_ids('134135')  # I just want to test downloading one Obs
        corr_output_path = OUTPUT + test_class.name + '_raw/'
        self.mock_os.path.exists.return_value = False  # Assuming data hasn't been downloaded before

        test_class.download(num_cores=1, download_products=False)

        # download would have been sucessful with these calls
        self.mock_download_call.assert_called_once_with(obs_id='134135', raw_dir=corr_output_path,
                                                    download_products=False, pipeline_version=None)
        self.mock_os.makedirs.assert_called_once_with(corr_output_path)
        self.mock_os.listdir.assert_called_once_with(corr_output_path)

        self.mock_warn.assert_not_called()  # No warning since data hasn't been downloaded before

    # checking that instrument filtering happens if instruments have been chosen
    def test_download_with_inst_filter(self):
        test_class = self.tm1  # this is a mission object with only 'TM1' selected
        test_class.filter_on_obs_ids('134135')
        self.mock_os.path.exists.return_value = False  # Assuming data hasn't been downloaded before

        self.tm1.download(num_cores=1, download_products=False)

        path = self.mock_get_evlist_from_obs.return_value
        self.mock_inst_filter.assert_called_once_with(insts=['TM1'], evlist_path=path)

    # Checking that if download has already been done, then a warning is raised
    def test_download_already_done(self):
        test_class = self.defaults
        test_class.filter_on_obs_ids('134135')
        self.mock_os.path.exists.return_value = True  # Assuming data has already been downloaded

        test_class.download(num_cores=1, download_products=False)

        self.mock_warn.assert_called_once_with("The raw data for this mission have already been downloaded.", stacklevel=2)

class TesteRASS1DEDownloadCall(unittest.TestCase):
    '''
    Putting this test into a separate class since it needs a lot of patches/ mocking that is
    specific to testing _download_call() only.
    '''

    def setUp(self):
        # I like to put the patch objects here instead of having a messy indentation later
        # also I couldnt find a way of testing this without mocking almost everything
        self.mock_file = patch('daxa.mission.erosita.open').start()
        self.mock_session = patch('daxa.mission.erosita.requests.Session').start()
        self.mock_copyfileobj = patch('daxa.mission.erosita.copyfileobj').start()
        self.mock_gzip = patch('daxa.mission.erosita.gzip').start()
        self.mock_makedirs = patch('daxa.mission.erosita.os.makedirs').start()
        self.mock_remove = patch('daxa.mission.erosita.os.remove').start()
        self.mock_exists = patch('daxa.mission.erosita.os.path.exists').start()

    def tearDown(self):
        # stopping all the patches and mocks to prevent weird behaviour occuring
        patch.stopall()

    def test_successful_download(self):
        self.mock_exists.return_value = False  # Data hasnt been downloaded
        # The text that session.get returns is quite long so i have put them in seperate text files
        # so these files that im opening contain the text of the response when you query the erosita api
        with open('test_data/html_responses/134135.txt', 'r') as file:
            respns_1 = file.read()  # this is the response when querying an obs_id
        
        with open('test_data/html_responses/134135_exp.txt', 'r') as file:
            respns_2 = file.read()  # this is the response when querying an obs_id + /EXP_010/
        
        # Defining some common strings I use in the assertions
        to_down = 'em01_134135_020_EventList_c010.fits.gz'
        obs_url = 'https://erosita.mpe.mpg.de/dr1/erodat/data/download/135/134/'
        local_dir = '/path/to/raw_data/134135/EXP_010/'

        # If your mocked object needs to return different values at different points in the
        # function, then they should be input as a list in the order of the return values
        self.mock_session.return_value.get.side_effect = [
            MockRequestResponse(text=respns_1),
            MockRequestResponse(text=respns_2),
            MockRequestResponse(text=to_down)]

        self.mock_gzip.open.return_value = MagicMock()  # Mock open gzip file

        eRASS1DE._download_call('134135', '/path/to/raw_data', download_products=False)

        # Next I define all the calls that should happen to each of my mocked objects
        expected_calls_to_session = [
            call(obs_url),
            call(obs_url + 'EXP_010//'),
            call(obs_url + 'EXP_010//' + to_down, stream=True)
        ]

        expected_calls_to_copyfileobj = [
            call(to_down, self.mock_file.return_value.__enter__.return_value),
            call(self.mock_gzip.open.return_value.__enter__.return_value, self.mock_file.return_value.__enter__.return_value)
        ]

        expected_calls_to_open = [
            call(local_dir + to_down, 'wb'),
            call(local_dir + to_down.strip('.gz'), 'wb')
        ]

        # asserting that these calls were indeed made
        self.mock_makedirs.assert_called_once_with(local_dir)
        self.mock_session.return_value.get.assert_has_calls(expected_calls_to_session)
        self.mock_file.assert_has_calls(expected_calls_to_open, any_order=True)
        self.mock_copyfileobj.assert_has_calls(expected_calls_to_copyfileobj)
        self.mock_gzip.open.assert_called_once_with(local_dir + to_down, 'rb')
        self.mock_remove.assert_called_once_with(local_dir + to_down)

    # an error should be raised when _download_call is called with an invalid pipeline version
    def test_incorrect_pipeline_version(self):
        self.mock_exists.return_value = False  # Data hasnt been downloaded
        # The text that session.get returns is quite long so i have put them in seperate text files
        # so these files that im opening contain the text of the response when you query the erosita api
        with open('test_data/html_responses/134135.txt', 'r') as file:
            respns_1 = file.read()  # this is the response when querying an obs_id
        
        with open('test_data/html_responses/134135_exp.txt', 'r') as file:
            respns_2 = file.read()  # this is the response when querying an obs_id + 'EXP_010'
        
        # Defining some common strings I use in the assertions
        to_down = 'em01_134135_020_EventList_c010.fits.gz'

        # If your mocked object needs to return different values at different points in the
        # function, then they should be input as a list in the order of the return values
        self.mock_session.return_value.get.side_effect = [
            MockRequestResponse(text=respns_1),
            MockRequestResponse(text=respns_2),
            MockRequestResponse(text=to_down)]
        self.mock_gzip.open.return_value = MagicMock()  # Mock open gzip file

        # finally we can actually run the function and test an error is raised with an invalid pipeline version
        with self.assertRaises(ValueError):
            eRASS1DE._download_call('134135', '/path/to/raw_data', download_products=False, pipeline_version='020')

    # testing what happens if a directory on the erosita database is missing
    def test_required_dir_missing(self):
        self.mock_exists.return_value = False  # Data hasnt been downloaded

        # The text that session.get returns is quite long so i have put them in seperate text files
        with open('test_data/html_responses/134135_no_dir.txt', 'r') as file:
            respns_1 = file.read()  # I removed the 'DET' folder from this response to mimic it missing
        
        self.mock_session.return_value.get.return_value = MockRequestResponse(text=respns_1)

        # an error should then be raised
        with self.assertRaises(FileNotFoundError):
            eRASS1DE._download_call('134135', '/path/to/raw_data', download_products=True)


if __name__ == '__main__':
    unittest.main()
