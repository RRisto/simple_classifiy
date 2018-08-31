import tempfile, os, unittest, shutil

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from classify.ClassifierCv import ClassifierCv
from tests.test_data.texts import Texts


class TestClassifierCv(unittest.TestCase):

    def setUp(self):
        df = Texts.df

        # countvectorizer needs Series type of data
        texts = pd.Series((text for text in df['text']))
        self.texts = texts
        self.labels=df['class']

        #temp dir
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_initalization(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        self.assertEqual(type(cf_cv.labels), pd.core.series.Series)
        self.assertEqual(type(cf_cv.labels_bin), np.ndarray)
        self.assertEqual(len(cf_cv.labels_unique),2)
        self.assertEqual(type(cf_cv.text),pd.core.series.Series)
        self.assertGreater(len(cf_cv.text),2)

    def test_prepare_pipeline(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        pipeline=[('tfidf', TfidfTransformer()),
                  ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                        alpha=1e-3, random_state=42,
                                        max_iter=5, tol=None)),]
        cf_cv.prepare_pipeline(pipeline)
        self.assertEqual(cf_cv.text_clf._final_estimator.loss, 'hinge')
        self.assertEqual(cf_cv.text_clf._final_estimator.max_iter, 5)
        self.assertEqual(cf_cv.text_clf._final_estimator.penalty, 'l2')
        self.assertEqual(cf_cv.text_clf._final_estimator.random_state, 42)

    def test_random_search(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        cf_cv.prepare_pipeline([('vect', CountVectorizer()),
                                ('tfidf', TfidfTransformer()),
                                ('crf', MultinomialNB())])

        param_dist = {'crf__alpha': [0.01, 0.02, 0.05, 0.07, 0.09]}

        cf_cv.perform_random_search(param_dist)

        params=cf_cv.get_top_random_search_parameters(1)
        self.assertEqual(type(params), dict)
        self.assertEqual(len(params), 1)

    def test_prepare_cv(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        cf_cv.prepare_pipeline([('vect', CountVectorizer()),
                                ('tfidf', TfidfTransformer()),
                                ('crf', MultinomialNB())])
        cf_cv.prepare_cv(3)
        self.assertEqual(cf_cv.kf.n_splits,3)
        self.assertIsNotNone(cf_cv.unique_labels)

    def test_train_save_metrics(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        name='MultinomialNB'
        metric='f1'
        cf_cv.train_save_metrics([('vect', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', MultinomialNB(alpha=.05)), ],
                                 metric, name,
                                 self.test_dir,
                                 self.test_dir)

        self.assertTrue(os.path.isfile(os.path.join(self.test_dir,name+'_'+metric+'.png')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir,name+'ROC_AUC.png')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir,name+'prec_recall.png')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir,name+'.xlsx')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir,name+'_average.xlsx')))
        self.assertEqual(type(cf_cv.roc_auc), dict)
        self.assertEqual(type(cf_cv.tpr), dict)
        self.assertEqual(type(cf_cv.fpr), dict)
        self.assertEqual(type(cf_cv.metrics_average_df), pd.DataFrame)
        self.assertEqual(type(cf_cv.metrics_df), pd.DataFrame)
        self.assertEqual(type(cf_cv.metrics_per_class), list)
        self.assertEqual(type(cf_cv.metrics_average), list)

    def test_get_one_metric(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        name = 'MultinomialNB'
        metric = 'f1'
        cf_cv.train_save_metrics([('vect', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', MultinomialNB(alpha=.05)), ],
                                 metric, name,
                                 self.test_dir,
                                 self.test_dir)

        f1=cf_cv.get_one_metric_cv('f1')
        precision=cf_cv.get_one_metric_cv('precision')
        recall=cf_cv.get_one_metric_cv('recall')
        support=cf_cv.get_one_metric_cv('support')

        f1_average = cf_cv.get_one_metric_cv('f1', average=True)
        precision_average = cf_cv.get_one_metric_cv('precision', average=True)
        recall_average = cf_cv.get_one_metric_cv('recall', average=True)
        support_average = cf_cv.get_one_metric_cv('support', average=True)

        self.assertEqual(type(f1), pd.DataFrame)
        self.assertEqual(type(precision), pd.DataFrame)
        self.assertEqual(type(recall), pd.DataFrame)
        self.assertEqual(type(support), pd.DataFrame)

        self.assertEqual(type(support_average), pd.DataFrame)
        self.assertEqual(type(recall_average), pd.DataFrame)
        self.assertEqual(type(precision_average), pd.DataFrame)
        self.assertEqual(type(f1_average), pd.DataFrame)

    def test_make_average_auc_boxplot(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        name = 'MultinomialNB'
        metric = 'f1'
        cf_cv.train_save_metrics([('vect', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', MultinomialNB(alpha=.05)), ],
                                 metric, name,
                                 self.test_dir,
                                 self.test_dir)

        filename=os.path.join(self.test_dir,name+'roc_auc_boxplot.png')
        cf_cv.make_average_auc_boxplot(savefile=filename)
        self.assertTrue(os.path.isfile(filename))

