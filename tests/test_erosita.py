import unittest
from unittest.mock import patch, MagicMock, call
from requests import Session
import numpy as np
from numpy.testing import assert_array_equal
import os
from io import BytesIO
import shutil


from astropy.units import Quantity
from astropy.io import fits

from daxa.mission import eRASS1DE, eROSITACalPV
from daxa import OUTPUT
from daxa.config import EROSITA_CALPV_INFO
from daxa.exceptions import DAXADownloadError


class TesteROSITACalPV(unittest.TestCase):
    def setUp(self):
        self.defaults = eROSITACalPV()
        self.filtered = eROSITACalPV(fields='eFEDS')
        self.alt_field_nme = eROSITACalPV(fields='crab iii')
        self.crab = eROSITACalPV(fields='crab')
        self.field_type = eROSITACalPV(fields='survey')
        self.type_n_nme = eROSITACalPV(fields=['survey', 'puppis a'])
    
    def test_chosen_fields(self):

        self.assertEqual(self.defaults.chosen_fields,list(set(EROSITA_CALPV_INFO["Field_Name"].tolist())))

        self.filtered = eROSITACalPV(fields='eFEDS')
        self.assertEqual(self.filtered.chosen_fields, ['EFEDS'])
        # TODO check why this fails
        #assert_array_equal(self.filtered.filtered_obs_ids, np.array(['300007', '300008', '300009', '300010']))

        # can't pass fields with the wrong type
        with self.assertRaises(ValueError):
            eROSITACalPV(fields=7)
        
        # can't pass a list if all elements arent strings
        with self.assertRaises(ValueError):
            eROSITACalPV(fields=['ok', 'ok', 7])
        
        # Incorrect fields raise an error
        with self.assertRaises(ValueError) as err:
            eROSITACalPV(fields=['efeds', 'eta cha', 'nope'])
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

    def test_filter_on_fields(self):
        self.defaults.filter_on_fields('efeds')
        self.assertEqual(self.defaults.chosen_fields, ['EFEDS'])
        assert_array_equal(self.defaults.filtered_obs_ids, np.array(['300007', '300008', '300009', '300010']))
    
    def test_name(self):
        self.assertEqual(self.defaults.name, 'erosita_calpv')
    
    def test_id_regex(self):
        self.assertEqual(self.defaults.id_regex, '^[0-9]{6}$')
    
    def test_fov(self):
        with self.assertWarns(UserWarning):
            self.assertEqual(self.defaults.fov, Quantity(30, 'arcmin'))
    
    def test_filter_on_obs_ids(self):
        with self.assertWarns(UserWarning):
            self.defaults.filter_on_obs_ids('700195')
            assert_array_equal(self.defaults.filtered_obs_ids, np.array(['700199', '700200']))


    def test_download_call_now(self):
        # for some reason this is only working in a context manager but not using decorators, i havent got the foggiest why
        with patch('daxa.mission.erosita.requests.get') as mock_p:
            with patch('daxa.mission.erosita.tarfile.open') as mock_t:
                mock_response = MagicMock()
                mock_response.raw = BytesIO(b'fake_data')
                mock_p.return_value.__enter__.return_value = mock_response
            
                print(mock_response.raw)
                mock_tarfile = MagicMock()
                mock_tarfile.extractcall = 'doesntmatter'
                mock_t.return_value.open.return_value.__enter__.return_value = mock_tarfile

                link = 'https://erosita.mpe.mpg.de/edr/eROSITAObservations/CalPvObs/eta_Cha.tar.gz'

                eROSITACalPV._download_call('test_data', link)


        shutil.rmtree('test_data/temp_download')



class TestERASS1DE(unittest.TestCase):
    def setUp(self):
        self.defaults = eRASS1DE()
        self.tm1 = eRASS1DE(insts='TM1')

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
        # Test fits file of structure: col1 [1, 2, 3, 4, 5] = evts 
        # col2 [1, 2, 5, 1, 7] = TM_NR
        evlist_path = 'test_data/inst_filt.fits' 

        eRASS1DE._inst_filtering(insts, evlist_path)

        self.assertTrue(os.path.exists('test_data/inst_filt_if_12.fits'))

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
        mock_exists.return_value = True

        eRASS1DE._inst_filtering(insts, evlist_path)

        mock_fits.open.assert_not_called()


class TesteRASS1DEDownload(unittest.TestCase):
    def setUp(self):
        self.defaults = eRASS1DE()
        self.tm1 = eRASS1DE(insts='TM1')

        self.mock_download_call = patch.object(eRASS1DE, '_download_call').start()
        self.mock_get_evlist_from_obs = patch.object(eRASS1DE, 'get_evlist_path_from_obs').start()
        self.mock_inst_filter = patch.object(eRASS1DE, '_inst_filtering').start()

        self.mock_session = patch('daxa.mission.erosita.requests.Session').start()
        self.mock_open = patch('builtins.open').start()
        self.mock_os = patch('daxa.mission.erosita.os').start()
        self.mock_tqdm = patch('daxa.mission.erosita.tqdm').start()
        self.mock_Pool = patch('daxa.mission.erosita.Pool').start()
        self.mock_warn = patch('daxa.mission.erosita.warn').start()
        self.mock_DAXADownloadError = patch('daxa.mission.erosita.DAXADownloadError').start()
    
    def tearDown(self):
        patch.stopall()
        self.mock_download_call.stop()
        self.mock_get_evlist_from_obs.stop()
        self.mock_inst_filter.stop()

    def test_download(self):
        test_class = self.defaults
        test_class.filter_on_obs_ids('134135')
        corr_output_path = OUTPUT + test_class.name + '_raw/'
        self.mock_os.path.exists.return_value = False  # Assuming data hasn't been downloaded before


        # Mocking _download_call method
        test_class.download(num_cores=1, download_products=False)
        self.mock_download_call.assert_called_once_with(obs_id='134135', raw_dir=corr_output_path,
                                                    download_products=False, pipeline_version=None)

        self.mock_os.makedirs.assert_called_once_with(corr_output_path)
        self.mock_os.listdir.assert_called_once_with(corr_output_path)

        self.mock_warn.assert_not_called()  # No warning since data hasn't been downloaded before
    

    def test_download_with_inst_filter(self):
        test_class = self.tm1
        test_class.filter_on_obs_ids('134135')
        self.mock_os.path.exists.return_value = False  # Assuming data hasn't been downloaded before

        # Mocking _download_call method
        self.tm1.download(num_cores=1, download_products=False)
        path = self.mock_get_evlist_from_obs.return_value
        self.mock_inst_filter.assert_called_once_with(insts=['TM1'], evlist_path=path)


    def test_download_already_done(self):
        test_class = self.defaults
        test_class.filter_on_obs_ids('134135')
        self.mock_os.path.exists.return_value = True  # Assuming data has already been downloaded

        test_class.download(num_cores=1, download_products=False)

        self.mock_warn.assert_called_once_with("The raw data for this mission have already been downloaded.", stacklevel=2)

    '''
    def test_download_error_handling(self):
        test_class = self.defaults
        test_class.filter_on_obs_ids('134135')

        self.mock_os.path.exists.return_value = False  # Assuming data hasn't been downloaded before
        
        # Force the condition where raised_errors is not 0 by setting it to a non-empty list
        with patch.object(test_class, '_download_call') as mock_download_call:
            mock_download_call.side_effect = ValueError('download error')
            test_class.download(num_cores=2, download_products=False)

        self.mock_DAXADownloadError.assert_called_once_with("['download error']")
    '''

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

# I tested this function in a different class because it needed so many unique patches
class TesteRASS1DEDownloadCall(unittest.TestCase):
    '''
    Tests the internal _download_call method of eRASS1DE
    '''

    def setUp(self):
        # I like to put the patch objects here instead of having a messy indentation later
        self.mock_file = patch('daxa.mission.erosita.open').start()
        self.mock_session = patch('daxa.mission.erosita.requests.Session').start()
        self.mock_copyfileobj = patch('daxa.mission.erosita.copyfileobj').start()
        self.mock_gzip = patch('daxa.mission.erosita.gzip').start()
        self.mock_makedirs = patch('daxa.mission.erosita.os.makedirs').start()
        self.mock_remove = patch('daxa.mission.erosita.os.remove').start()
        self.mock_exists = patch('daxa.mission.erosita.os.path.exists').start()

    def tearDown(self):
        patch.stopall()

    def test_successful_download(self):
        self.mock_exists.return_value = False  # Data hasnt been downloaded
        # The text that session.get returns is quite long so i have put them in seperate text files
        with open('test_data/html_responses/134135.txt', 'r') as file:
            respns_1 = file.read()
        
        with open('test_data/html_responses/134135_exp.txt', 'r') as file:
            respns_2 = file.read()
        
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

        self.mock_makedirs.assert_called_once_with(local_dir)
        self.mock_session.return_value.get.assert_has_calls(expected_calls_to_session)
        self.mock_file.assert_has_calls(expected_calls_to_open, any_order=True)
        self.mock_copyfileobj.assert_has_calls(expected_calls_to_copyfileobj)
        self.mock_gzip.open.assert_called_once_with(local_dir + to_down, 'rb')
        self.mock_remove.assert_called_once_with(local_dir + to_down)
    
    def test_sucessful_product_download(self):
        self.mock_exists.return_value = False  # Data hasnt been downloaded
        # The text that session.get returns is quite long so i have put them in seperate text files
        with open('test_data/html_responses/134135.txt', 'r') as file:
            respns_1 = file.read()
        with open('test_data/html_responses/134135_exp.txt', 'r') as file:
            respns_2 = file.read()
        with open('test_data/html_responses/134135_det.txt', 'r') as file:
            respns_3 = file.read()

        # Defining some common strings I use in the assertions
        to_down = ['em01_134135_020_EventList_c010.fits.gz',
                    'em01_134135_021_Image_c010.fits.gz',
                    'em01_134135_022_Image_c010.fits.gz',
                    'em01_134135_023_Image_c010.fits.gz',
                    'em01_134135_024_Image_c010.fits.gz',
                    'em01_134135_025_Image_c010.fits.gz',
                    'em01_134135_026_Image_c010.fits.gz',
                    'em01_134135_027_Image_c010.fits.gz',
                    'em01_134135_021_ExposureMap_c010.fits.gz',
                    'em01_134135_022_ExposureMap_c010.fits.gz',
                    'em01_134135_023_ExposureMap_c010.fits.gz',
                    'em01_134135_024_ExposureMap_c010.fits.gz',
                    'em01_134135_025_ExposureMap_c010.fits.gz',
                    'em01_134135_026_ExposureMap_c010.fits.gz',
                    'em01_134135_027_ExposureMap_c010.fits.gz']

        obs_url = 'https://erosita.mpe.mpg.de/dr1/erodat/data/download/135/134/'
        local_dir_exp = '/path/to/raw_data/134135/EXP_010/'
        local_dir_det = '/path/to/raw_data/134135/DET_010/'


        # If your mocked object needs to return different values at different points in the
        # function, then they should be input as a list in the order of the return values
        self.mock_session.return_value.get.side_effect = [
            MockRequestResponse(text=respns_1), 
            MockRequestResponse(text=respns_2), 
            MockRequestResponse(text=to_down[0]),
            MockRequestResponse(text=to_down[1]),
            MockRequestResponse(text=to_down[2]),
            MockRequestResponse(text=to_down[3]),
            MockRequestResponse(text=to_down[4]),
            MockRequestResponse(text=to_down[5]),
            MockRequestResponse(text=to_down[6]),
            MockRequestResponse(text=to_down[7]),
            MockRequestResponse(text=respns_3),
            MockRequestResponse(text=to_down[8]),
            MockRequestResponse(text=to_down[9]),
            MockRequestResponse(text=to_down[10]),
            MockRequestResponse(text=to_down[11]),
            MockRequestResponse(text=to_down[12]),
            MockRequestResponse(text=to_down[13]),
            MockRequestResponse(text=to_down[14])
            ]

        self.mock_gzip.open.return_value = MagicMock()  # Mock open gzip file

        eRASS1DE._download_call('134135', '/path/to/raw_data', download_products=True)

        expected_calls_to_makedirs = [
            call(local_dir_exp),
            call(local_dir_det)
        ]

        # Im sure the following can be done in a neater list comphrehendsion
        expected_calls_to_session = [
            call(obs_url),
            call(obs_url + 'EXP_010//'),
            call(obs_url + 'EXP_010//' + to_down[0], stream=True),
            call(obs_url + 'EXP_010//' + to_down[1], stream=True),
            call(obs_url + 'EXP_010//' + to_down[2], stream=True),
            call(obs_url + 'EXP_010//' + to_down[3], stream=True),
            call(obs_url + 'EXP_010//' + to_down[4], stream=True),
            call(obs_url + 'EXP_010//' + to_down[5], stream=True),
            call(obs_url + 'EXP_010//' + to_down[6], stream=True),
            call(obs_url + 'EXP_010//' + to_down[7], stream=True),
            call(obs_url + 'DET_010//'),
            call(obs_url + 'DET_010//' + to_down[8], stream=True),
            call(obs_url + 'DET_010//' + to_down[9], stream=True),
            call(obs_url + 'DET_010//' + to_down[10], stream=True),
            call(obs_url + 'DET_010//' + to_down[11], stream=True),
            call(obs_url + 'DET_010//' + to_down[12], stream=True),
            call(obs_url + 'DET_010//' + to_down[13], stream=True),
            call(obs_url + 'DET_010//' + to_down[14], stream=True),
        ]

        expected_calls_to_copyfileobj = [
            call(to_down[0], self.mock_file.return_value.__enter__.return_value),
            call(self.mock_gzip.open.return_value.__enter__.return_value, self.mock_file.return_value.__enter__.return_value),
            call(to_down[1], self.mock_file.return_value.__enter__.return_value),
            call(self.mock_gzip.open.return_value.__enter__.return_value, self.mock_file.return_value.__enter__.return_value),
            call(to_down[2], self.mock_file.return_value.__enter__.return_value),
            call(self.mock_gzip.open.return_value.__enter__.return_value, self.mock_file.return_value.__enter__.return_value),
            call(to_down[3], self.mock_file.return_value.__enter__.return_value),
            call(self.mock_gzip.open.return_value.__enter__.return_value, self.mock_file.return_value.__enter__.return_value),
            call(to_down[4], self.mock_file.return_value.__enter__.return_value),
            call(self.mock_gzip.open.return_value.__enter__.return_value, self.mock_file.return_value.__enter__.return_value),
            call(to_down[5], self.mock_file.return_value.__enter__.return_value),
            call(self.mock_gzip.open.return_value.__enter__.return_value, self.mock_file.return_value.__enter__.return_value),
            call(to_down[6], self.mock_file.return_value.__enter__.return_value),
            call(self.mock_gzip.open.return_value.__enter__.return_value, self.mock_file.return_value.__enter__.return_value),
            call(to_down[7], self.mock_file.return_value.__enter__.return_value),
            call(self.mock_gzip.open.return_value.__enter__.return_value, self.mock_file.return_value.__enter__.return_value),
            call(to_down[8], self.mock_file.return_value.__enter__.return_value),
            call(self.mock_gzip.open.return_value.__enter__.return_value, self.mock_file.return_value.__enter__.return_value),
            call(to_down[9], self.mock_file.return_value.__enter__.return_value),
            call(self.mock_gzip.open.return_value.__enter__.return_value, self.mock_file.return_value.__enter__.return_value),
            call(to_down[10], self.mock_file.return_value.__enter__.return_value),
            call(self.mock_gzip.open.return_value.__enter__.return_value, self.mock_file.return_value.__enter__.return_value),
            call(to_down[11], self.mock_file.return_value.__enter__.return_value),
            call(self.mock_gzip.open.return_value.__enter__.return_value, self.mock_file.return_value.__enter__.return_value),
            call(to_down[12], self.mock_file.return_value.__enter__.return_value),
            call(self.mock_gzip.open.return_value.__enter__.return_value, self.mock_file.return_value.__enter__.return_value),
            call(to_down[13], self.mock_file.return_value.__enter__.return_value),
            call(self.mock_gzip.open.return_value.__enter__.return_value, self.mock_file.return_value.__enter__.return_value)
        ]

        expected_calls_to_open = [
            call(local_dir_exp + to_down[0], 'wb'), 
            call(local_dir_exp + to_down[0].strip('.gz'), 'wb'),
            call(local_dir_exp + to_down[1], 'wb'), 
            call(local_dir_exp + to_down[1].strip('.gz'), 'wb'),
            call(local_dir_exp + to_down[2], 'wb'), 
            call(local_dir_exp + to_down[2].strip('.gz'), 'wb'),
            call(local_dir_exp + to_down[3], 'wb'), 
            call(local_dir_exp + to_down[3].strip('.gz'), 'wb'),
            call(local_dir_exp + to_down[4], 'wb'), 
            call(local_dir_exp + to_down[4].strip('.gz'), 'wb'),
            call(local_dir_exp + to_down[5], 'wb'), 
            call(local_dir_exp + to_down[5].strip('.gz'), 'wb'),
            call(local_dir_exp + to_down[6], 'wb'), 
            call(local_dir_exp + to_down[6].strip('.gz'), 'wb'),
            call(local_dir_exp + to_down[7], 'wb'), 
            call(local_dir_exp + to_down[7].strip('.gz'), 'wb'),
            call(local_dir_det + to_down[8], 'wb'), 
            call(local_dir_det + to_down[8].strip('.gz'), 'wb'),
            call(local_dir_det + to_down[9], 'wb'), 
            call(local_dir_det + to_down[9].strip('.gz'), 'wb'),
            call(local_dir_det + to_down[10], 'wb'), 
            call(local_dir_det + to_down[10].strip('.gz'), 'wb'),
            call(local_dir_det + to_down[11], 'wb'), 
            call(local_dir_det + to_down[11].strip('.gz'), 'wb'),
            call(local_dir_det + to_down[12], 'wb'), 
            call(local_dir_det + to_down[12].strip('.gz'), 'wb'),
            call(local_dir_det + to_down[13], 'wb'), 
            call(local_dir_det + to_down[13].strip('.gz'), 'wb')
            ]
        
        expected_calls_to_gzipopen = [
            call(local_dir_exp + down, 'rb') for down in to_down if not 'Exp' in down
        ] + [call(local_dir_det + down, 'rb') for down in to_down if 'Exp' in down]

        expected_calls_to_osremove = [
            call(local_dir_exp + down) for down in to_down if not 'Exp' in down
        ] + [call(local_dir_det + down) for down in to_down if 'Exp' in down]


        self.mock_makedirs.assert_has_calls(expected_calls_to_makedirs)
        self.mock_session.return_value.get.assert_has_calls(expected_calls_to_session)
        self.mock_file.assert_has_calls(expected_calls_to_open, any_order=True)
        self.mock_copyfileobj.assert_has_calls(expected_calls_to_copyfileobj)
        self.mock_gzip.open.assert_has_calls(expected_calls_to_gzipopen, any_order=True)
        self.mock_remove.assert_has_calls(expected_calls_to_osremove)


    def test_incorrect_pipeline_version(self):
        self.mock_exists.return_value = False  # Data hasnt been downloaded
        # The text that session.get returns is quite long so i have put them in seperate text files
        with open('test_data/html_responses/134135.txt', 'r') as file:
            respns_1 = file.read()
        
        with open('test_data/html_responses/134135_exp.txt', 'r') as file:
            respns_2 = file.read()
        
        # Defining some common strings I use in the assertions
        to_down = 'em01_134135_020_EventList_c010.fits.gz'

        # If your mocked object needs to return different values at different points in the
        # function, then they should be input as a list in the order of the return values
        self.mock_session.return_value.get.side_effect = [
            MockRequestResponse(text=respns_1), 
            MockRequestResponse(text=respns_2), 
            MockRequestResponse(text=to_down)]
        self.mock_gzip.open.return_value = MagicMock()  # Mock open gzip file

        with self.assertRaises(ValueError):
            eRASS1DE._download_call('134135', '/path/to/raw_data', download_products=False, pipeline_version='020')

    def test_required_dir_missing(self):
        self.mock_exists.return_value = False  # Data hasnt been downloaded

        # The text that session.get returns is quite long so i have put them in seperate text files
        with open('test_data/html_responses/134135_no_dir.txt', 'r') as file:
            # this text file has the DET directories removed
            respns_1 = file.read()
        
        self.mock_session.return_value.get.return_value = MockRequestResponse(text=respns_1)

        with self.assertRaises(FileNotFoundError):
            eRASS1DE._download_call('134135', '/path/to/raw_data', download_products=True)



if __name__ == '__main__':
    unittest.main()
