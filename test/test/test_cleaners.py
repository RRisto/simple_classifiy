import unittest, os
from TextClass.Cleaners import Cleaners


class TestCleaners(unittest.TestCase):
    def test_initalization_with_stopwords_from_list(self):
        stopwords=['a','the']
        cl = Cleaners(stopwords)
        self.assertEqual(cl.stopwords, stopwords)

    def test_initalization_without_stopwords(self):
        cl = Cleaners(stopwords=None)
        self.assertIsNone(cl.stopwords)
        self.assertIsNone(cl.stopwords)

    def test_initialization_with_stopwords_from_file(self):
        stopwords_filename='test/stopwords/stopwords.txt'
        cl = Cleaners(stopwords_filename)

        with open(stopwords_filename) as file:
            stopwords = file.readlines()
        stopwords_list = []
        for line in stopwords:
            stopwords_list.extend(line.strip().split(','))

        stopwords_list = list(set(stopwords_list))
        stopwords_list.remove('')
        stopwords_list = [stopword.strip() for stopword in stopwords_list]

        self.assertEqual(cl.stopwords, stopwords_list)

    def test_load_stopwords(self):
        cl = Cleaners(None)
        cl._load_stopwords(os.path.join(os.getcwd(), 'test/stopwords/stopwords.txt'))
        self.assertEqual(type(cl.stopwords), list)
        self.assertEqual(len(cl.stopwords), 2)

    def test_tokenize(self):
        cl = Cleaners()
        tokens = cl.tokenize('mina olen pikk poiss')
        self.assertEqual(tokens, ['mina', 'olen', 'pikk', 'poiss'])

    def test_tokenize_list(self):
        cl = Cleaners()
        tokens = cl.tokenize([['mina olen'], ['pikk poiss']])
        self.assertEqual(tokens, [['mina', 'olen'], ['pikk', 'poiss']])

    def test_remove_stopwords_return_list(self):
        cl = Cleaners()
        clean = cl.remove_stopwords(['mina', 'olen', 'pikk', 'poiss'], stopwords=['olen'])
        self.assertEqual(clean, ['mina', 'pikk', 'poiss'])

    def test_remove_stopwords_return_string(self):
        cl = Cleaners()
        clean = cl.remove_stopwords(['mina', 'olen', 'pikk', 'poiss'], stopwords=['olen'], return_string=True)
        self.assertEqual(clean, 'mina pikk poiss')

    def test_remove_stopwords_default(self):
        cl = Cleaners('test/stopwords/stopwords.txt')
        clean = cl.remove_stopwords(['yes', 'very', 'no'], return_string=True)
        self.assertEqual(clean, 'very')

    def test_lower(self):
        cl = Cleaners()
        text = cl.lower_text('Suur TEXT')
        self.assertEqual(text, 'suur text')

    def test_remove_punctuation_no_custom(self):
        cl = Cleaners()
        no_punct = cl.remove_punctuation('palju:;" on siin!')
        self.assertEqual(no_punct, 'palju    on siin ')

    def test_remove_punctuation_custom(self):
        cl = Cleaners()
        no_punct = cl.remove_punctuation('suured 4 ja 5 pikad laused', custom_punctutation='45')
        self.assertEqual(no_punct, 'suured   ja   pikad laused')

    def test_remove_excess_space(self):
        cl = Cleaners()
        clean = cl.remove_excess_spaces('many   spaces    here')
        self.assertEqual(clean, 'many spaces here')

    def test_replace_regex_pattern(self):
        cl = Cleaners()
        clean = cl.replace_regex_pattern(pattern='\d', text='minu tekst 5 on siin', replace='')
        self.assertEqual(clean, 'minu tekst  on siin')

    def test_replace_regex_pattern_list(self):
        cl = Cleaners()
        clean = cl.replace_regex_pattern(pattern='\d', text=['minu', 'tekst','5', 'on siin'], replace='')
        self.assertEqual(clean,  ['minu', 'tekst', '', 'on siin'])

    def test_replace_regex_pattern_literally(self):
        cl=Cleaners()
        text='verygood \d it is'
        clean=cl.replace_regex_pattern('\d', text, escape_regex=True)
        self.assertEqual(clean, 'verygood   it is')

    def test_stem_as_list(self):
        cl=Cleaners()
        text='I was flying around heavily'
        text_stemmed=cl.stem(text, 'english')
        self.assertEqual(text_stemmed, ['i', 'was', 'fli', 'around', 'heavili'])

    def test_stem_as_str(self):
        cl=Cleaners()
        text='I was flying around heavily'
        text_stemmed=cl.stem(text, 'english', as_list=False)
        self.assertEqual(text_stemmed, 'i was fli around heavili')

    def test_replace_string_from_list(self):
        cl=Cleaners()
        text="very bad movie it was"
        clean=cl.replace_string_from_list(text,['movie'])
        self.assertEqual(clean, "very bad   it was")

