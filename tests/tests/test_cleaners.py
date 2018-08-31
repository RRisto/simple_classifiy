import unittest
from classify.Cleaners import Cleaners
import os

class TestCleaners(unittest.TestCase):

    def test_initalization_with_stopwords(self):
        cl=Cleaners()
        self.assertEqual(cl.stopwords, None)

    def test_initalization_without_stopwords(self):
        cl=Cleaners(stopwords=None)
        self.assertIsNone(cl.stopwords)
        self.assertIsNone(cl.stopwords)

    def test_load_stopwords(self):
        cl=Cleaners(None)
        cl._load_stopwords(os.path.join(os.getcwd(),'tests/test_data/stopwords.txt'))
        self.assertEqual(type(cl.stopwords), list)
        self.assertEqual(len(cl.stopwords), 2)

    def test_lemmatize_et_result_string(self):
        cl=Cleaners()
        lemmas=cl.lemmatize_et('Väga suured mehed olid')
        self.assertEqual(lemmas, 'väga suur mees olema')

    def test_lemmatize_et_result_list(self):
        cl=Cleaners()
        lemmas=cl.lemmatize_et('Väga suured mehed olid', as_list=True)
        self.assertEqual(lemmas, ['väga', 'suur', 'mees', 'olema'])

    def test_tokenize_et(self):
        cl=Cleaners()
        tokens=cl.tokenize('mina olen pikk poiss')
        self.assertEqual(tokens, ['mina', 'olen', 'pikk', 'poiss'])

    def test_remove_stopwords_return_list(self):
        cl=Cleaners()
        clean=cl.remove_stopwords(['mina', 'olen', 'pikk', 'poiss'], stopwords=['olen'])
        self.assertEqual(clean, ['mina', 'pikk', 'poiss'])

    def test_remove_stopwords_return_string(self):
        cl=Cleaners()
        clean=cl.remove_stopwords(['mina', 'olen', 'pikk', 'poiss'], stopwords=['olen'], return_string=True)
        self.assertEqual(clean, 'mina pikk poiss')

    def test_lower(self):
        cl=Cleaners()
        text=cl.lower_text('Suur TEXT')
        self.assertEqual(text, 'suur text')

    def test_remove_punctuation_no_custom(self):
        cl=Cleaners()
        no_punct=cl.remove_punctuation('palju:;" on siin!')
        self.assertEqual(no_punct, 'palju    on siin ')

    def test_remove_punctuation_custom(self):
        cl=Cleaners()
        no_punct=cl.remove_punctuation('suured 4 ja 5 pikad laused', custom_punctutation='45')
        self.assertEqual(no_punct, 'suured   ja   pikad laused')

    def test_remove_excess_space(self):
        cl=Cleaners()
        clean=cl.remove_excess_spaces('many   spaces    here')
        self.assertEqual(clean, 'many spaces here')

    def test_replace_regex_pattern(self):
        cl=Cleaners()
        clean=cl.replace_regex_pattern(pattern='\d', text='minu tekst 5 on siin',replace= '')
        self.assertEqual(clean, 'minu tekst  on siin')