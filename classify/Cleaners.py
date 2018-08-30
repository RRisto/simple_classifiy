import re, string
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

from estnltk import Text


class Cleaners(object):
    """class for collecting cleaning methods"""

    def __init__(self, stopwords=None):
        if isinstance(stopwords, list):
            self.stopwords=stopwords
        elif  isinstance(stopwords, str):
            self._load_stopwords(self.stopwords)
        else:
            self.stopwords=None

    def _load_stopwords(self, filename):
        """loads stopwords from file and clean them to list"""
        with open(filename) as file:
            stopwords=file.readlines()

        stopwords_list=[]
        for line in stopwords:
            stopwords_list.extend(line.strip().split(','))

        #remove duplicates
        stopwords_list=list(set(stopwords_list))
        stopwords_list.remove('')
        stopwords_list=[stopword.strip() for stopword in stopwords_list]
        self.stopwords=filename
        self.stopwords= stopwords_list


    def stem(self, text, lang, as_list=False):
        """stems text using porter stemmer from nltk

                Parameters
                ----------
                text : string/list of strings of text to be stemmed
                lang: string, language in which text is
                as_list : boolean, return results as stemmed token string (True) or concatenate it one string?

                Returns
                -------
                list of strings of lemmatized text
                """
        stemmer = SnowballStemmer(lang)
        if type(text) is not list:
            text=text.split(' ')

        stemmed_text=[]
        for word in text:
            stemmed_text.append(stemmer.stem(word))

        return stemmed_text

    def lemmatize_et(self, text, as_list=False):
        """lemmatizes text using estnltk lemmatizer

        Parameters
        ----------
        text : string of text to be lemmatized
        as_list : boolean, return results as lemmatized token string (True) or concatenate it one string?

        Returns
        -------
        string / list of lemmatized text
        """

        #needed for Text to work, otherwise after newline/tab no text is analyzed
        text=text.replace('\n', ' ')
        text=text.replace('\t', ' ')

        text=Text(text)
        text.analyse('morphology')
        lemmas=text.lemma
        word_forms=text.form
        lemmatized_text=[]

        for i,lemma in enumerate(lemmas):
            if word_forms[i][0]=='neg o' and 'olema' in lemmas[i]:
                lemmatized_text.append('pole')
            else:
                lemmatized_text.append(lemma[0])

        if as_list is False:
            lemmatized_text= ' '.join(lemmatized_text)

        return lemmatized_text


    def tokenize(self, text, tokenizer='estnltk'):
        """tokenize text using Estnltk or nltk

             Parameters
             ----------
             text : string of text to be tokenized
             lang : string, whic tokenizer to use, default 'estnltk', other 'nltk'

             Returns
             -------
             list of strings (tokens)"""
        if tokenizer=='esntltk':
            text = Text(text)
            text.analyse('morphology')
            return text.words.text
        #nltk tokenizer
        return word_tokenize(text)


    def remove_stopwords(self, tokens, stopwords=None, return_string=False, min_len_tokens_kept=3):
        """remove stopwords from token list

         Parameters
         ----------
         tokens : list of tokens where stopwrods are to be removed
         stopwords: list of stopwords. If this is None, takes default stopwords
         return_string: instead of list return string
         min_len_tokens_kept: minimum length of tokens kept

         Returns
         -------
         list/string of tokens without stopwords"""

        if stopwords is None:
            stopwords=self.stopwords

        result=[token for token in tokens if token not in stopwords and len(token)>=min_len_tokens_kept]

        if return_string:
            return ' '.join(result)

        return result

    def lower_text(self, text):
        """lowercase text

        Parameters
        ----------
        text : string of text to be turned lowercase

        Returns
        -------
        string of text text in lowercase"""

        return text.lower()


    def remove_punctuation(self, text, custom_punctutation=None, replace_with=" "):
        """remove punctuation from text

         Parameters
        ----------
        text : string of text where punctuation is to be removed

        custom_punctutation: string of custom punctutations to remove.
         If None default punctuation is removed

        replace_with: string with what punctuation to replace with

        Returns
        -------
        string of text without punctutation """

        punctuation=string.punctuation

        if custom_punctutation is not None:
            punctuation=punctuation+custom_punctutation

        return "".join(char if char not in punctuation else replace_with for char in text)

    def remove_excess_spaces(self, text):
        """removes excess spaces if more than one is in a row:
        tere  olen -> tere olen

          Parameters
        ----------
        text : string of text where excess spaces is to be removed

        Returns
        -------
        string of text without excess space
        """
        return re.sub(' +',' ', text)

    def replace_regex_pattern(self, pattern, text, replace=" ", escape_regex=False):
        """replaces regex patter in text with replace

         Parameters
        ----------
        text : string/list of strings of text where excess spaces is to be removed
        patter: regex pattern which is replaced
        replace: string with what pattern is to be replaced

        Returns
        -------
        string of text (or list of strings if inut was list of strings) with regex pattern replaced
        """
        if escape_regex:
            regex_pattern = re.escape(pattern)
        else:
            regex_pattern = re.compile(pattern)
        if type(text) is list:
            result=[]
            for el in text:
                result.append(re.sub(regex_pattern, replace, str(el)))
            return result

        return re.sub(regex_pattern, replace, str(text))

    def replace_string_from_list(self, text, string_list, replace=' '):
        """replaces string in text with replace

         Parameters
        ----------
        text : string/list of strings of text where strings from stringlist is to be replaced
        string_list: list of strings to be replaced
        replace: string with what pattern is to be replaced

        Returns
        -------
        string of text where string_list is replaced
        """
        text = str(text)
        for strng in string_list:
            text = text.replace(strng, replace)
        return text

