import unittest

from mock import patch
from test.test_classifiercv import TestClassifierCv
from test.test_fasttext_classifier import TestFasttextClassifier
from test.test_ensemble_classifier import TestEnsembleClassifier
from test.test_cleaners import TestCleaners
from test.test_lda import TestLda

@patch("matplotlib.pyplot.show")  # not to pop up plots
def runtests(mock_fig):
    unittest.main()

if __name__ == '__main__':
    runtests()

