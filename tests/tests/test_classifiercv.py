import tempfile, os, unittest, shutil

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from TextClass.ClassifierCv import ClassifierCv
from tests.test_data.texts import Texts


class TestClassifierCv(unittest.TestCase):
    def setUp(self):
        df_train = Texts.df_classification_train
        self.df_test=Texts.df_classification_test

        # countvectorizer needs Series type of data
        texts = pd.Series((text for text in df_train['text']))
        self.texts = texts
        self.labels = df_train['class']

        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_initalization(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        self.assertEqual(type(cf_cv.labels), pd.core.series.Series)
        self.assertEqual(type(cf_cv.labels_bin), np.ndarray)
        self.assertEqual(len(cf_cv.labels_unique), 2)
        self.assertEqual(type(cf_cv.text), pd.core.series.Series)
        self.assertGreater(len(cf_cv.text), 2)

    def test_prepare_pipeline(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        pipeline = [('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                          alpha=1e-3, random_state=42,
                                          max_iter=5, tol=None)), ]
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

        params = cf_cv.get_top_random_search_parameters(1)
        params_print=cf_cv.print_top_random_search(1)

        self.assertEqual(type(params), dict)
        self.assertEqual(len(params), 1)
        self.assertEqual(params_print, None)

    def test_prepare_cv(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        cf_cv.prepare_pipeline([('vect', CountVectorizer()),
                                ('tfidf', TfidfTransformer()),
                                ('crf', MultinomialNB())])
        cf_cv.prepare_cv(3)
        self.assertEqual(cf_cv.kf.n_splits, 3)
        self.assertIsNotNone(cf_cv.unique_labels)

    def test_train_save_metrics(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        name = 'MultinomialNB'
        metric = 'f1'
        cf_cv.train_save_metrics([('vect', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', MultinomialNB(alpha=.05)), ],
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

    def test_train_save_metrics_no_paths(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        name = 'MultinomialNB'
        metric = 'f1'
        cf_cv.train_save_metrics([('vect', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', MultinomialNB(alpha=.05)), ],
                                 metric, name,
                                None,
                                 None)

        self.assertFalse(os.path.isfile(os.path.join(self.test_dir, name + '_' + metric + '.png')))
        self.assertFalse(os.path.isfile(os.path.join(self.test_dir, name + 'ROC_AUC.png')))
        self.assertFalse(os.path.isfile(os.path.join(self.test_dir, name + 'prec_recall.png')))
        self.assertFalse(os.path.isfile(os.path.join(self.test_dir, name + '.xlsx')))
        self.assertFalse(os.path.isfile(os.path.join(self.test_dir, name + '_average.xlsx')))
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

        f1 = cf_cv.get_one_metric_cv('f1')
        precision = cf_cv.get_one_metric_cv('precision')
        recall = cf_cv.get_one_metric_cv('recall')
        support = cf_cv.get_one_metric_cv('support')

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

        filename = os.path.join(self.test_dir, name + 'roc_auc_boxplot.png')
        cf_cv.make_average_auc_boxplot(savefile=filename)
        self.assertTrue(os.path.isfile(filename))

    def test_make_metric_boxplot(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        name = 'MultinomialNB'
        metric = 'f1'
        cf_cv.train_save_metrics([('vect', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', MultinomialNB(alpha=.05))],
                                 metric, name,
                                 self.test_dir,
                                 self.test_dir)

        filename = os.path.join(self.test_dir, name + '_metric_boxplot.png')
        cf_cv.make_metric_boxplot(metric='f1',savefile=filename)
        self.assertTrue(os.path.isfile(filename))

    def test_make_roc_auc_plot(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        name = 'MultinomialNB'
        metric = 'f1'
        cf_cv.train_save_metrics([('vect', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', MultinomialNB(alpha=.05))],
                                 metric, name,
                                 self.test_dir,
                                 self.test_dir)

        filename = os.path.join(self.test_dir, name + '_roc_auc_plot.png')
        cf_cv.make_roc_auc_plot(savefile=filename)
        self.assertTrue(os.path.isfile(filename))

    def test_plot_confusion_matrix(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        name = 'MultinomialNB'
        metric = 'f1'
        cf_cv.train_save_metrics([('vect', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', MultinomialNB(alpha=.05)), ],
                                 metric, name,
                                 self.test_dir,
                                 self.test_dir)

        filename = os.path.join(self.test_dir, name + '_confusion_matrix.png')
        cf_cv.plot_confusion_matrix(savefile=filename)
        self.assertTrue(os.path.isfile(filename))

    def test_plot_confusion_matrix_eval_data_normalize(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        name = 'MultinomialNB'
        metric = 'f1'
        cf_cv.train_save_metrics([('vect', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', MultinomialNB(alpha=.05)), ],
                                 metric, name,
                                 self.test_dir,
                                 self.test_dir)

        filename = os.path.join(self.test_dir, name + 'eval_confusion_matrix.png')
        cf_cv.labels_eval_real=['pos','neg','neg','neg']
        cf_cv.labels_eval_predicted=['pos','neg','neg','neg']

        cf_cv.plot_confusion_matrix(savefile=filename,normalize=True,use_evaluation_data=True)
        self.assertTrue(os.path.isfile(filename))

    def test_calc_evaluation_report(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        name = 'MultinomialNB'
        metric = 'f1'
        cf_cv.train_save_metrics([('vect', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', MultinomialNB(alpha=.05))],
                                 metric, name,
                                 self.test_dir,
                                 self.test_dir)

        filename = os.path.join(self.test_dir, '_eval_report')
        cf_cv.calc_evaluation_report(self.df_test['text'], self.df_test['class'], savefile=filename)
        self.assertTrue(os.path.isfile(filename + "_" + name + ".csv"))
        self.assertTrue(os.path.isfile(filename + "_" + name + "_average.csv"))

    def test_predict_labels(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        name = 'MultinomialNB'
        metric = 'f1'
        cf_cv.train_save_metrics([('vect', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', MultinomialNB(alpha=.05)), ],
                                 metric, name,
                                 self.test_dir,
                                 self.test_dir)
        labels = cf_cv.predict(['bad', 'good'])
        self.assertTrue(all(labels == ['neg', 'pos']))

    def test_predict_labels_probas(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        name = 'MultinomialNB'
        metric = 'f1'
        cf_cv.train_save_metrics([('vect', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', MultinomialNB(alpha=.05)), ],
                                 metric, name,
                                 self.test_dir,
                                 self.test_dir)
        labels = cf_cv.predict(['bad', 'good'], proba=True)
        self.assertTrue(labels.shape == (2, 2))
        self.assertEqual(type(labels), pd.DataFrame)
        self.assertEqual(len(labels['pos'].values), 2)
        self.assertEqual(len(labels['neg'].values), 2)
        self.assertEqual(type(labels['neg'].values[0]), np.float64)

    def test_pickle(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        name = 'MultinomialNB'
        metric = 'f1'
        cf_cv.train_save_metrics([('vect', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', MultinomialNB(alpha=.05))],
                                 metric, name,
                                 self.test_dir,
                                 self.test_dir)
        savefile = os.path.join(self.test_dir, 'clf_cv.cv')
        cf_cv.pickle(savefile)
        self.assertTrue(os.path.isfile(savefile))

    def test_unpickle(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        name = 'MultinomialNB'
        metric = 'f1'
        cf_cv.train_save_metrics([('vect', CountVectorizer()),
                                  ('tfidf', TfidfTransformer()),
                                  ('clf', MultinomialNB(alpha=.05))],
                                 metric, name,
                                 self.test_dir,
                                 self.test_dir)
        savefile = os.path.join(self.test_dir, 'clf_cv.cv')
        cf_cv.pickle(savefile)
        new_cf_cv = ClassifierCv.unpickle(savefile)

        texts = ['dont know that', 'nice good and bad']
        predicted_labels_orig = cf_cv.predict(texts, proba=True)
        predicted_labesl_new = new_cf_cv.predict(texts, proba=True)

        self.assertTrue(all(predicted_labels_orig == predicted_labesl_new))
