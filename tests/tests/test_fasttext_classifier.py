import tempfile, unittest, shutil, os

import pandas as pd

from classify.ClassifierCv import ClassifierCv
from classify.FasttextClassifier import FasttextClassifier
from tests.test_data.texts import Texts


class TestFasttextClassifier(unittest.TestCase):
    def setUp(self):
        df = Texts.df_classification_train
        self.df_test = Texts.df_classification_test

        # countvectorizer needs Series type of data
        texts = pd.Series((text for text in df['text']))
        self.texts = texts
        self.labels = df['class']

        # temp dir
        self.test_dir = tempfile.mkdtemp()
        self.output = os.path.join(self.test_dir, 'model.ft')

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_init(self):
        ft_clf = FasttextClassifier()
        self.assertEqual(ft_clf.lr, 0.1)
        self.assertEqual(ft_clf.lr_update_rate, 100)
        self.assertEqual(ft_clf.dim, 100)
        self.assertEqual(ft_clf.ws, 5)
        self.assertEqual(ft_clf.epoch, 100)

    def test_fit(self):
        ft_clf = FasttextClassifier(output=self.output)
        ft_clf.fit(self.texts, self.labels)
        self.assertEqual(ft_clf.classes_, list(self.labels.unique()))
        self.assertTrue(all(ft_clf.x == self.texts))
        self.assertTrue(all(ft_clf.y == self.labels))
        self.assertTrue(os.path.isfile(self.output + '.bin'))

    def test_predict(self):
        ft_clf = FasttextClassifier(output=self.output)
        ft_clf.fit(self.texts, self.labels)
        labels = ft_clf.predict(['very bad', 'very good'])
        self.assertTrue(all(labels == ['neg', 'pos']))

    def test_predict_wrong_type(self):
        ft_clf = FasttextClassifier(output=self.output)
        ft_clf.fit(self.texts, self.labels)
        labels = ft_clf.predict("text")
        self.assertEqual(labels, None)

    def test_predict_proba(self):
        ft_clf = FasttextClassifier(output=self.output)
        ft_clf.fit(self.texts, self.labels)
        probas = ft_clf.predict_proba(['very bad', 'very good'])
        self.assertTrue(probas[0][0] < probas[0][1])
        self.assertTrue(probas[1][0] > probas[1][1])

    def test_predict_proba_wrong_type(self):
        ft_clf = FasttextClassifier(output=self.output)
        ft_clf.fit(self.texts, self.labels)
        probas = ft_clf.predict_proba('text')
        self.assertEqual(probas, None)

    def test_load_pretrained(self):
        ft_clf = FasttextClassifier(output=self.output)
        ft_clf.fit(self.texts, self.labels)
        loaded_ft_clf = FasttextClassifier()
        loaded_ft_clf.loadpretrained(self.output + '.bin')
        labels = loaded_ft_clf.predict(['very bad', 'very good'])
        self.assertTrue(all(labels == ['neg', 'pos']))

    def test_xft_classifiercv(self):
        cf_cv = ClassifierCv(self.labels, self.texts)
        name = 'ft'
        metric = 'f1'
        cf_cv.train_save_metrics([('clf', FasttextClassifier(output=self.output, epoch=1))],
                                 metric, name,
                                 self.test_dir,
                                 self.test_dir)

        filename = os.path.join(self.test_dir, '_eval_report')
        cf_cv.calc_evaluation_report(self.df_test['text'], self.df_test['class'], savefile=filename)
        self.assertTrue(os.path.isfile(filename + "_" + name + ".csv"))
        self.assertTrue(os.path.isfile(filename + "_" + name + "_average.csv"))
