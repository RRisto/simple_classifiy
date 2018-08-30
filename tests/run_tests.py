import unittest

from tests.test_classifiercv import TestClassifierCv
from tests.test_cleaners import TestCleaners
from tests.test_lda import TestLda


def create_suite():
    test_suite = unittest.TestSuite()
    test_suite.addTest(TestClassifierCv())
    test_suite.addTest(TestLda())
    test_suite.addTest(TestCleaners())
    return test_suite

if __name__ == '__main__':
   suite = create_suite()
   #
   suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestClassifierCv)
   unittest.TextTestRunner().run(suite)
   # runner=unittest.TextTestRunner()
   # runner.run(suite)

