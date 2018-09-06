import os
import shutil
import tempfile
import unittest

import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from TextClass.ClassifierCv import ClassifierCv
from TextClass.EnsembleClassifier import EnsembleClassifier
from TextClass.FasttextClassifier import FasttextClassifier
from test_data.texts import Texts


class TestEnsembleClassifier(unittest.TestCase):
    def setUp(self):
        df = Texts.df_classification_train

        # countvectorizer needs Series type of data
        texts = pd.Series((text for text in df['text']))
        self.texts = texts
        self.labels = df['class']

        self.test_dir = tempfile.mkdtemp()
        self.ft_output = output = os.path.join(self.test_dir, 'model.ft')

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_ft_noweights(self):
        clf1 = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                           ('clf', LogisticRegression())])
        clf2 = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                           ('clf', MultinomialNB())])
        clf3 = FasttextClassifier(epoch=2, output=self.ft_output)
        eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3])

        cf_cv = ClassifierCv(self.labels, self.texts)
        name = 'ensemble_clf'
        metric = 'f1'
        cf_cv.train_save_metrics([('clf', eclf)],
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
        self.assertEqual(type(cf_cv.metrics_per_class), list)
        self.assertEqual(type(cf_cv.metrics_average), list)

    def test_ft_weights(self):
        clf1 = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                           ('clf', LogisticRegression())])
        clf2 = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                           ('clf', MultinomialNB())])
        clf3 = FasttextClassifier(epoch=2, output=self.ft_output)
        eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], weights=[1,1,0.5])

        cf_cv = ClassifierCv(self.labels, self.texts)
        name = 'ensemble_clf'
        metric = 'f1'
        cf_cv.train_save_metrics([('clf', eclf)],
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
        self.assertEqual(type(cf_cv.metrics_per_class), list)
        self.assertEqual(type(cf_cv.metrics_average), list)
