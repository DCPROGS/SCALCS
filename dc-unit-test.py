#! /usr/bin/env python

import unittest
from scalcs.test import test as tt

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(tt.TestDC_PyPs)
    unittest.TextTestRunner(verbosity=2).run(suite)
