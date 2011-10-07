#! /usr/bin/env python

import dctest.test as tt
import unittest

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(tt.TestDC_PyPs)
    unittest.TextTestRunner(verbosity=2).run(suite)
