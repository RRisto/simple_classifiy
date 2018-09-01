import unittest

from mock import patch
from tests.test_classifiercv import TestClassifierCv
# from tests.test_fasttext_classifier import TestFasttextClassifier
# from tests.test_cleaners import TestCleaners
# from tests.test_lda import TestLda


@patch("matplotlib.pyplot.show") #not to pop up plots
def runtests(mock_fig):
   unittest.main()

if __name__ == '__main__':
   runtests()
   # unittest.main()

