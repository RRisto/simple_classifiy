import tempfile, unittest, shutil
import pandas as pd

from classify.FasttextClassifier import FasttextClassifier
from tests.test_data.texts import Texts


class TestFasttextClassifier(unittest.TestCase):

    def setUp(self):
        df = Texts.df

        # countvectorizer needs Series type of data
        texts = pd.Series((text for text in df['text']))
        self.texts = texts
        self.labels = df['class']

        # temp dir
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_init(self):
        ft_clf=FasttextClassifier()
        self.assertEqual(ft_clf.lr, 0.1)
        self.assertEqual(ft_clf.lr_update_rate , 100)
        self.assertEqual(ft_clf.dim, 100)
        self.assertEqual(ft_clf.ws, 5)
        self.assertEqual(ft_clf.epoch, 100)

    def test_fit(self):
        ft_clf = FasttextClassifier()
        ft_clf.fit(self.texts, self.labels)
        self.assertEqual(ft_clf.classes_, list(self.labels.unique()))
        self.assertTrue(all(ft_clf.x==self.texts))
        self.assertTrue(all(ft_clf.y==self.labels))

    def test_predict(self):
        ft_clf = FasttextClassifier()
        ft_clf.fit(self.texts, self.labels)
        labels=ft_clf.predict(['very bad', 'very good'])
        self.assertTrue(all(labels, ['']))