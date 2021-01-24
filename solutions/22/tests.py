
import unittest


from .solution import main



class Tests(unittest.TestCase):
    def test_given_input_returns_expected_result(self):
        given_input = [(30,75),(0,50),(60,150)]
        expected_result = 2
        self.assertEquals(main(given_input),expected_result)
    
if __name__ == "__main__":
    unittest.main()