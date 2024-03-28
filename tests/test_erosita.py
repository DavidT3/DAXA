import unittest

from daxa.mission import eRASS1DE

class TestErosita(unittest.TestCase):
    
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



if __name__ == '__main__':
    unittest.main()
