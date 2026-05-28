import os
import unittest

from scalcs import scalcsio

class TestMecLoad(unittest.TestCase):
      
    def test_load_from_excel(self):
        filename = './samples/samples.xlsx'
        os.path.isfile(filename)
        self.mec, mectitle = scalcsio.load_from_excel_sheet(filename, sheet=0)
        assert len(self.mec.States) == 5
        assert len(self.mec.Rates) == 10
        assert len(self.mec.Cycles) == 1

    def test_mr(self):
        filename = './samples/samples.xlsx'
        mec, mectitle = scalcsio.load_from_excel_sheet(filename, sheet=1)
        assert mec.Rates[8].rateconstants == 3.0
        assert mec.Rates[10].rateconstants == 2.0
        mec.update_mr()
        assert mec.Rates[8].rateconstants == 1.0
        assert mec.Rates[10].rateconstants == 1.0

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMecLoad)
    unittest.TextTestRunner(verbosity=2).run(suite)