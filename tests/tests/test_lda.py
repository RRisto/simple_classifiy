import unittest, os, shutil, tempfile

import numpy as np

from classify.LDA import CustomLda
from tests.test_data.texts import Texts


class TestLda(unittest.TestCase):
    def setUp(self):
        self.text = Texts.df_lda
        # temp dir
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_lda_init(self):
        lda = CustomLda(self.text)
        self.assertEqual(type(lda.data), list)
        self.assertEqual(type(lda.corpus), list)
        self.assertGreater(len(lda.dictionary), 1)
        self.assertGreater(len(lda.data), 1)
        self.assertGreater(len(lda.corpus), 1)

    def test_train(self):
        lda = CustomLda(self.text)
        lda.train(2)
        self.assertEqual(lda.num_topics, 2)

    def test_get_topic(self):
        lda = CustomLda(self.text)
        lda.train(2)
        topics = lda.get_topics(2)
        self.assertEqual(type(topics), list)
        self.assertGreater(len(topics), 1)

    def test_get_topic_terms(self):
        lda = CustomLda(self.text)
        lda.train(2)
        topics = lda.get_topic_terms(1)
        self.assertEqual(type(topics), list)
        self.assertGreater(len(topics), 1)

    def test_get_perplexity(self):
        lda = CustomLda(self.text)
        lda.train(2)
        perp = lda.get_preplexity()
        self.assertEqual(type(perp), np.float64)

    def test_save_ldavis(self):
        lda = CustomLda(self.text)
        lda.train(2)
        filename = os.path.join(self.test_dir, 'test_ldavis.html')
        lda.save_ldavis(filename)
        self.assertTrue(os.path.isfile(filename))

    def test_save_lda(self):
        lda = CustomLda(self.text)
        lda.train(2)
        filename = os.path.join(self.test_dir, 'model.lda')
        lda.save_lda(filename)
        self.assertTrue(os.path.isfile(filename))

    def test_pickle_unpickle(self):
        lda = CustomLda(self.text)
        lda.train(2)
        filename = os.path.join(self.test_dir, 'model.pickle')
        lda.pickle(filename)
        lda2 = CustomLda().unpickle(filename)
        self.assertEqual(lda.corpus, lda2.corpus)
        self.assertEqual(lda.data, lda2.data)
        self.assertEqual(lda.dictionary, lda2.dictionary)
