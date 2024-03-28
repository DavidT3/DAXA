import unittest

from daxa.mission import eRASS1DE

class TestErosita(unittest.TestCase):
    
    def setUp(self):

        self.mission = eRASS1DE()
    
    def test_chosen_instruments(self):

        # error is raised for invalid instruments
        with self.assertRaises(ValueError):
            eRASS1DE(insts='wrong')


if __name__ == '__main__':
    unittest.main()
