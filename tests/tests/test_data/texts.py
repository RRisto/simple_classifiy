import pandas as pd


class Texts:
    df=pd.DataFrame({'text':['very good','nice and clean', 'very clean', 'very nice','nice and good',
                             'very good', 'nice and clean', 'very clean', 'very nice', 'nice and good',
                             'very bad', 'terrible and ugly','very bad', 'very ugly','bad and ugly',
                             'very bad', 'terrible and ugly', 'very bad', 'very ugly', 'bad and ugly'],
                     'class':['pos','pos','pos','pos','pos','pos','pos','pos','pos','pos',
                              'neg','neg','neg','neg','neg', 'neg','neg','neg','neg','neg']})

    df_lda=pd.DataFrame({'tokens':[['very','good'],['nice', 'and', 'some', 'other', 'text'],
                                   ['good','text']]})