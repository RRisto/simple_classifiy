import pickle
import sys, os
import tempfile

import numpy as np
import pandas as pd
import fasttext as ft

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report



class FasttextClassifier(BaseEstimator, ClassifierMixin):
    """wrapper for sklearn. Base classifer of Fasttext estimator"""

    def __init__(self, lpr='__label__', lr=0.1, lr_update_rate =100, dim=100, ws=5, epoch=100, minc=1, neg=5, ngram=1,
                 loss='softmax', nbucket=0, minn=0, maxn=0, thread=4, silent=0, output="model"):
        """
        label_prefix   			label prefix ['__label__']
        lr             			learning rate [0.1]
        lr_update_rate 			change the rate of updates for the learning rate [100]
        dim            			size of word vectors [100]
        ws             			size of the context window [5]
        epoch          			number of epochs [5]
        min_count      			minimal number of word occurences [1]
        neg            			number of negatives sampled [5]
        word_ngrams    			max length of word ngram [1]
        loss           			loss function {ns, hs, softmax} [softmax]
        bucket         			number of buckets [0]
        minn           			min length of char ngram [0]
        maxn           			min length of char ngram [0]
        todo : Recheck need of some of the variables, present in default classifier
        """

        self.label_prefix = lpr
        self.lr = lr
        self.lr_update_rate = lr_update_rate
        self.dim = dim
        self.ws = ws
        self.epoch = epoch
        self.min_count = minc
        self.neg = neg
        self.word_ngrams = ngram
        self.loss = loss
        self.bucket = nbucket
        self.minn = minn
        self.maxn = maxn
        self.thread = thread
        self.silent = silent
        self.classifier = None
        self.result = None
        self.original_array=None #if oneVsRest classifier input this will flag
        self.classes=None #oneVsRest classifier
        self.output = output
        self.lpr = lpr
        self.classes_=None

    def fit(self, x, y):
        """
        Input:
            -x: list, array of texts for training
            -y: list, array of labels for training
        OUTPUT:
            -returns classifier object
        """
        self.x=x
        self.y=y

        #make temp folder to keep data, ft special issue
        temp_dir=tempfile.mkdtemp()

        df = pd.DataFrame({'class': self.y, 'text': self.x})
        self.classes_=list(df['class'].unique())

        #if y is numpy array, it is from OneVsRest claasifier is calling and
        ##  have to do some magic here, turn array into string for ft
        if type(y) is np.ndarray:
            self.original_array=y
            df['class'] = '__label__'+df['class'].astype(str)+' '
            self.classes=df['class'].unique()
        else:
            # just in case if new call is made for same object but input is not array
            self.original_array=None
            df['class'] = '__label__' + df['class'].astype(str) + ' '

        temp_file=os.path.join(temp_dir, 'data_train.csv')
        df.to_csv(temp_file, index = False)

        self.classifier = ft.supervised(temp_file, self.output, dim=self.dim, lr=self.lr, lr_update_rate =self.lr_update_rate ,
                                        epoch=self.epoch, min_count=self.min_count, word_ngrams=self.word_ngrams,
                                        bucket=self.bucket, thread=self.thread, silent=self.silent, label_prefix=self.lpr)
        return (None)


    def predict(self, texts, k_best=1):
        """
        Input:
            - texts: array, list of texts for predicting
            - k_best: int, number of top labels returned
        OUTPUT:
            - list of predicted labels for texts
        """
        try:
            if (type(texts) is list or type(texts) is pd.core.series.Series):
                labels=self.classifier.predict(texts, k=k_best)
                #todo
                # result_temp=['__label__'+lbl[0]+' ' for lbl in labels]
                result_temp=[lbl[0] for lbl in labels]
                self.result=np.array(result_temp)
        except:
            print("Error in input dataset.. please see if the file/list of sentences is of correct format")
            sys.exit(-1)
        return (self.result)


    def predict_proba(self, texts, k_best=1):
        """
         Input:
            - texts: array, list of texts for predicting
            - k_best: int, number of top labels returned
        OUTPUT:
            - list of predicted labels for texts
        """
        try:
            if type(texts) is list or type(texts) is pd.core.series.Series:
                if self.original_array is not None: #oneVsRest classifier gave input case
                    self.result = self.classifier.predict_proba(texts, k=len(self.classes))
                    #now sort based on tuple first elements
                    result_temp=[sorted(lst, key=lambda x: x[0]) for lst in self.result]
                    result_mat=[]
                    #make matrix
                    for rw in result_temp:
                        temp=[]
                        for elem in rw:
                            temp.append(elem[1])
                        result_mat.append(temp)
                    self.result=np.array(result_mat)
                else:
                    self.result = self.classifier.predict_proba(texts, k=len(self.classes_))
                    # order lables based on lcasses order, turn into matrix
                    # classes_ordered = [label[:-1].replace(self.label_prefix, '') for label in self.classes_]
                    classes_ordered = self.classes_#[label[:-1].replace(self.label_prefix, '') for label in self.classes_]
                    result_mat = []
                    for row in self.result:
                        temp_row = []
                        # order row based on classes order and keep only proba
                        for cat in classes_ordered:
                            temp_row.extend([item[1] for item in row if item[0] == cat])
                        result_mat.append(temp_row)
                    self.result=np.array(result_mat)

        except:
            print("Error in input dataset.. please see if the file/list of sentences is of correct format")
            sys.exit(-1)

        return (self.result)


    def loadpretrained(self, filename):
        """
        INPUT:
            -filename: string, file from where to load pretrianed model
        OUTPUT:
            - initialises self.classifier
            """
        self.classifier = ft.load_model(filename, label_prefix=self.lpr)
