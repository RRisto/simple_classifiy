import tempfile
import unittest
import os
import pandas as pd
import shutil

from classify.ClassifierCv import ClassifierCv
from classify.Cleaners import Cleaners
from classify.FasttextClf import FastTextClassifier

os.chdir('..')
os.chdir('..')


class TestClassifierCv(unittest.TestCase):

    def setUp(self):
        df = pd.read_csv('tests_data/data.csv')
        df=df.head(20)
        cl = Cleaners()
        texts = []

        for text in df['text']:
            text = cl.remove_punctuation(text)
            text = cl.lower_text(text)
            tokens = cl.tokenize(text)
            tokens = cl.remove_stopwords(tokens, return_string=True)  # removes default stopwords
            texts.append(tokens)

        # countvectorizer needs Series type of data
        texts = pd.Series((text for text in texts))
        self.texts = texts
        self.labels=df['class']

        #temp dir
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_random_search(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        cf_cv.prepare_pipeline([('crf', FastTextClassifier())])

        param_dist = {'crf__loss': ['ns', 'hs', 'softmax']}

        cf_cv.perform_random_search(param_dist)

        params=cf_cv.get_top_random_search_parameters(1)
        self.assertEqual(type(params), dict)
        self.assertEqual(len(params), 1)

    def test_train_save_metrics(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        name = 'fasttext'
        metric = 'f1'
        cf_cv.train_save_metrics([('crf', FastTextClassifier())],
                                 metric, name,
                                 self.test_dir,
                                 self.test_dir)

        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, name + '_' + metric + '.png')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, name + 'ROC_AUC.png')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, name + 'prec_recall.png')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, name + '.xlsx')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, name + '_average.xlsx')))
        self.assertEqual(type(cf_cv.roc_auc), dict)
        self.assertEqual(type(cf_cv.tpr), dict)
        self.assertEqual(type(cf_cv.fpr), dict)
        self.assertEqual(type(cf_cv.metrics_average_df), pd.DataFrame)
        self.assertEqual(type(cf_cv.metrics_df), pd.DataFrame)
        self.assertEqual(type(cf_cv.metrics), list)
        self.assertEqual(type(cf_cv.metrics_average), list)