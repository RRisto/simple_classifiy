import string, os, itertools, pickle
from time import time

import pandas as pd
import numpy as np
from scipy import interp

from sklearn.preprocessing import LabelBinarizer,label_binarize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, precision_recall_curve, \
    average_precision_score, confusion_matrix, classification_report
# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt


class TwoLabelBinarizer(LabelBinarizer):
    """my label binarizer so that it would give same result in binary cases as in multiclass
    default binarizer turns out funny format when dealing with binary classification problem"""
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1-Y))
        else:
            return Y


class ClassifierCv(object):
    """class for general classifier"""

    def __init__(self, data_labels, data_text):
        self.text=data_text
        self.labels=data_labels

        if data_labels is not None: #should be none only if unpickle
            #turn into binary labels
            self.labels_unique=[label for label in self.labels.unique()]
            #for some reason in two classes label binareizer gives different output
            if len(self.labels_unique)==2:
                my_label_binarizer=TwoLabelBinarizer()
                self.labels_bin=my_label_binarizer.fit_transform(self.labels)
            else:
                self.labels_bin=label_binarize(self.labels, classes=self.labels_unique)
        else:
            self.labels_unique=None
            self.labels_bin=None

        #metrics (recall, prec, f1)
        self.metrics_per_class=None
        self.metrics_average=None
        #cv labels
        self.cv_labels_real=[]
        self.cv_labels_predicted=[]
        #roc auc
        self.fpr = None
        self.tpr = None
        self.roc_auc = None
        #precision-recall curve
        self.recall=None
        self.precision=None
        self.average_precision=None
        #needed for precison recall, keeps cv results
        self.y_real=None
        self.y_proba=None
        #grid search
        self.grid_search=None
        #time
        self.times_cv=[]
        self.time_train=[]


    def text_process(self, mess):
        """
        Default text cleaning. Takes in a string of text, then performs the following:
        1. Remove all punctuation
        2. Remove all stopwords
        3. Returns a list of the cleaned text
        """
        # Check characters to see if they are in punctuation
        nopunc = [char for char in mess if char not in string.punctuation]

        # Join the characters again to form the string.
        nopunc = ''.join(nopunc)

        # Now just remove any stopwords
        return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


    def prepare_pipeline(self, custom_pipeline=None):
        """prepares pipeline for model
        - INPUT:
            - custom_pipeline: Pipeline, if None, use default pipeline, else input list for sklearn Pipeline
        -OUTPUT:
            - initialises sklearn pipeline"""

        if custom_pipeline is None:
            self.text_clf = Pipeline([('vect', CountVectorizer(analyzer=self.text_process)),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                                       alpha=1e-3, random_state=42,
                                                       max_iter=5, tol=None)),
                                 ])
        else:
            self.text_clf=Pipeline(custom_pipeline)


    def perform_random_search(self, param_grid, scoring='f1_weighted',num_cv=3, n_jobs=1,**kwargs):
        """perform grid search to find best parameters
        -INPUT:
            - param_grid:  dict or list of dictionaries, Dictionary with parameters names (string) as keys and lists
             of parameter settings to try as values, or a list of such dictionaries, in which case the grids spanned
             by each dictionary in the list are explored. This enables searching over any sequence of parameter settings.
            - scoring: string from http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
            - num_cv: int, number of cross-validation iterations
            - n_jobs: Number of jobs to run in parallel.

        -OUTPUT:
            - fitted gridsearch"""

        self.grid_search = GridSearchCV(self.text_clf, cv=num_cv, scoring=scoring, n_jobs=n_jobs,
                                        param_grid=param_grid, **kwargs)
        self.grid_search.fit(self.text, self.labels)


    def print_top_random_search(self, num_top=3):
        """print grid search results
        -INPUT:
            -num_top: int, number of top search results to print
        -OUTPUT:
            - printed top results"""

        results=self.grid_search.cv_results_
        for i in range(1, num_top + 1):
            candidates = pd.np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")


    def get_top_random_search_parameters(self, num):
        """get parameters of top grid search
         -INPUT:
            - num: int, number of nth top rank parameters
        -OUTPUT:
            - dict of nth top parameters"""

        results = self.grid_search.cv_results_
        candidates = pd.np.flatnonzero(results['rank_test_score'] == num)
        for candidate in candidates:
            return results['params'][candidate]


    def prepare_cv(self, n_iter, shuffle=True, random_state=1):
        """initialises stratified cross-validaton
        INPUT:
            - n_iter: int, number of cross validation iterations
        OUTPUT:
            - prepares k-fold cross validation object"""

        self.kf = StratifiedKFold(n_splits=n_iter, shuffle=shuffle, random_state=random_state)
        self.unique_labels = list(self.labels.unique())


    def init_metrics_(self):
        """
        initialise metrics, remove previous training metrics
        """
        self.metrics_per_class = []
        self.metrics_average = []

        self.fpr = dict()
        self.tpr = dict()
        self.roc_auc = dict()

        self.precision = dict()
        self.recall = dict()
        self.average_precision = dict()
        self.y_proba = dict()
        self.y_real = dict()
        self.cv_labels_predicted = []
        self.cv_labels_real = []

        self.times_cv = []
        self.time_train = []

        for label_bin in range(len(self.labels_unique)):
            self.fpr[label_bin] = []
            self.tpr[label_bin] = []
            self.roc_auc[label_bin] = []
            self.precision[label_bin] = []
            self.recall[label_bin] = []
            self.average_precision[label_bin] = []
            self.y_real[label_bin] = []
            self.y_proba[label_bin] = []

        self.fpr["micro"] = []
        self.tpr["micro"] = []
        self.roc_auc["micro"] = []
        self.precision["micro"] = []
        self.recall["micro"] = []
        self.average_precision["micro"] = []
        self.y_real["micro"] = []
        self.y_proba["micro"] = []

    def calc_store_rocauc_precrec_(self, classifier_rocauc, proba_method, train_ids, test_ids):
        """calculate and store ROC AUC and precision recall curve metrics
        -INPUT:
            -classifier_roc_auc: sklearn OneVsRest classifier
            -proba_method: string, classifier method name for predicting label probability
            -train_ids: list of ids of samples used for training
            -test_ids: list of ids of samples used for testing
        -OUTPUT:
            -stored metrics for ROC AUC and precision recall curve
            """
        y_score = None
        # roc auc stuff
        # some classifiers have method decision function, others predict proba to get scores
        if proba_method == "decision_function":
            y_score = classifier_rocauc.fit(self.text[train_ids], self.labels_bin[train_ids]).decision_function(self.text[test_ids])
        elif proba_method == "predict_proba":
            y_score = classifier_rocauc.fit(self.text[train_ids], self.labels_bin[train_ids]).predict_proba(
                list(self.text[test_ids]))

        if y_score is None:
            return

        for i in range(len(self.unique_labels)):
            fpr_temp, tpr_temp, _ = roc_curve(self.labels_bin[test_ids][:, i], y_score[:, i])
            self.fpr[i].append(fpr_temp)
            self.tpr[i].append(tpr_temp)
            self.roc_auc[i].append(auc(fpr_temp, tpr_temp))
            # precison -recall metrics
            precision_temp, recall_temp, _ = precision_recall_curve(self.labels_bin[test_ids][:, i],
                                                                    y_score[:, i])
            self.precision[i].append(precision_temp)
            self.recall[i].append(recall_temp)
            self.average_precision[i].append(average_precision_score(self.labels_bin[test_ids][:, i],
                                                                     y_score[:, i]))
            self.y_real[i].append(self.labels_bin[test_ids][:, i])
            self.y_proba[i].append(y_score[:, i])

        # Compute micro-average ROC curve and ROC area
        fpr_micro_temp, tpr_micro_temp, _ = roc_curve(self.labels_bin[test_ids].ravel(), y_score.ravel())
        self.fpr["micro"].append(fpr_micro_temp)
        self.tpr["micro"].append(tpr_micro_temp)
        self.roc_auc["micro"].append(auc(fpr_micro_temp, tpr_micro_temp))

        # precision recall.  A "micro-average": quantifying score on all classes jointly
        prec_micro_temp, recall_micro_temp, _ = precision_recall_curve(self.labels_bin[test_ids].ravel(),
                                                                       y_score.ravel())
        self.precision["micro"].append(prec_micro_temp)
        self.recall["micro"].append(recall_micro_temp)
        self.average_precision["micro"] = average_precision_score(self.labels_bin[test_ids], y_score,
                                                                  average="micro")
        self.y_real["micro"].append(self.labels_bin[test_ids].ravel())
        self.y_proba["micro"].append(y_score.ravel())


    def get_classifier_proba_method_(self, classifier):
        """get label probability method of classifier. Some mehtods don't support predict_proba
        -INPUT:
            -classifier: sklearn classifier, which probability calculation method is to be detected
        -OUTPUT:
            -string with method name
            """
        proba_method = None

        if callable(getattr(classifier, "predict_proba", None)):
            proba_method = "predict_proba"
        elif callable(getattr(classifier, "decision_function", None)):
            proba_method = "decision_function"
        return proba_method


    def train(self, roc_auc=True):
        """train model, save metrics
        -INPUT:
            - roc_auc: boolean, should roc_auc (includeing precision -recall plot) metrics be saved

        _OUTPUT:
            - trained model with metrics"""

        self.init_metrics_()

        classifier_rocauc = OneVsRestClassifier(self.text_clf)

        #check if classifier has predict_proba or decison_function method
        proba_method=self.get_classifier_proba_method_(classifier_rocauc)

        for train, test in self.kf.split(self.text, self.labels):

            t0=time()

            self.text_clf.fit(self.text[train], self.labels[train])

            time_cv=time()-t0
            self.times_cv.append(time_cv)

            labels_predict = self.text_clf.predict(list(self.text[test]))
            self.cv_labels_predicted.append(labels_predict)
            self.cv_labels_real.append(self.labels[test])
            labels_predict_label=labels_predict

            # per class metric, not average
            self.metrics_per_class.append(precision_recall_fscore_support(self.labels[test],
                                                                          labels_predict_label,
                                                                          average=None,
                                                                          labels=self.unique_labels))

            self.metrics_average.append(precision_recall_fscore_support(self.labels[test],
                                                           labels_predict_label,
                                                           average='weighted',
                                                           labels=self.unique_labels))

            if roc_auc:
                self.calc_store_rocauc_precrec_(classifier_rocauc, proba_method, train, test)

        self.metrics_df = pd.DataFrame(self.metrics_per_class)
        self.metrics_average_df= pd.DataFrame(self.metrics_average)

        #finally make model with all training data
        t0=time()
        self.text_clf.fit(self.text, self.labels)
        time_train=time()-t0
        self.time_train.append(time_train)


    def predict(self, text_list, proba=False):
        """"predict labels based on trained classifier
        - INPUT:
            - text_list: list of texts which label will be predicted
            - proba: boolean, if true probability will be predicted
        - OUTPUT:
            - dataframe labels (with probas if proba True)
            """
        if proba:
            probas=[]
            if callable(getattr(self.text_clf, "predict_proba", None)):
                probas=self.text_clf.predict_proba(text_list)
            if callable(getattr(self.text_clf, "decision_function", None)):
                probas= self.text_clf.decision_function(text_list)
            return pd.DataFrame(probas, columns=self.unique_labels)

        return self.text_clf.predict(text_list)


    def get_one_metric_cv(self, metric_name, average=False):
        """"extract one metric from precision_recall_fscore_support to compare it between classes
        - INPUT:
            - metric_name: str, name of the metric to be extracted, on of the following:
                'precision', 'recall', 'f1', 'support'
            - average: boolean, True if data is average above all classes, else if per class: True
        - OUTPUT:
            - dataframe with metric from cross validation"""

        ind = 0
        if metric_name == 'precision':
            ind = 0
        if metric_name == 'recall':
            ind = 1
        if metric_name == 'f1':
            ind = 2
        if metric_name == 'support':
            ind = 3

        if average:
            return pd.DataFrame(self.metrics_average_df[ind].values.tolist()).transpose()
        return pd.DataFrame(self.metrics_df[ind].values.tolist(), columns=self.unique_labels).transpose()


    def make_metric_boxplot(self, metric, savefile=None, average=False, title=None, x_tick_rotation=45):
        """function to make metric boxplot to compare cross validation results between classes
        - INPUT:
            - metric: metric to be used for plot
            - savefile: path to file if plot is to be saved, else None
            - average: data used is average all classes (if False) or per class (True)
            - title: title of the plot, None if no title is to be used
        - OUTPUT:
            - plot of metric from cross validation results"""

        metric_df=self.get_one_metric_cv(metric, average)

        #print results
        print("MEDIAN")
        print(metric_df.median(axis=1))
        print('MEAN')
        print(metric_df.mean(axis=1))

        plt.boxplot(metric_df)

        if title is not None:
            plt.title(title)

        #add class labels if not average
        if not average:
            plt.xticks([i + 1 for i in range(len(self.unique_labels))], self.unique_labels,
                       rotation=x_tick_rotation)

        #set y-axis from 0 to 1
        axes = plt.gca()
        axes.set_ylim([0, 1])

        if savefile is not None:
            plt.savefig(savefile, bbox_inches='tight')


    def save_times(self, time_metrics_path, algorithm_name):
        """save times from training
        -INPUT:
            -time_metrics_path: str, path where to save time files
            - algorithm_name: str, name of the algorithm to add to file nime
        -OUTPUT:
            - save training times (from cv and final fit)"""
        df_time_cv=pd.DataFrame({'cv_times':self.times_cv})
        df_time_train=pd.DataFrame({'train_time':self.time_train})

        df_time_cv.to_csv(os.path.join(time_metrics_path, algorithm_name+'_time_cv.xlsx'), index=False)
        df_time_train.to_csv(os.path.join(time_metrics_path, algorithm_name+'_time_train.xlsx'), index=False)


    def train_save_metrics(self, pipeline, metric_name, algorithm_name, plot_path=None, metrics_path=None,
                           roc_auc_average=True, roc_auc=True, roc_auc_plot_cat_index=None, num_cv=10, random_state=1):
        """wrapper function to train model quickly and less code

        -INPUT:
            - pipeline: list, skleanr pipeline
            - metric_name: metric which plot is to be saved, must be one of these
                        'f1', 'precision' ,'recall', 'support'
            - algorithm_name: name of the algorithm used, uses this to plot titel and file name
            - plot_path: path of the plot to be saved
            - metrics_path: path of the metrics files to be saved
            - num_cv: number of cross validations
            - roc_auc_averge: boolean, plot average ROC AUC above all classes
            - roc_auc_plot_cat_index: int, integer of category which ROC_AUC to plot

        - OUTPUT:
            - plot of metric specified
            - ROC_AUC plot
            - metrics files (per class and average)"""

        self.prepare_pipeline(pipeline)
        self.prepare_cv(num_cv, random_state=random_state)
        self.train(roc_auc=roc_auc)
        self.algorithm_name=algorithm_name

        if plot_path is not None:
            metric_plot_path=os.path.join(plot_path, algorithm_name+'_'+ metric_name+ '.png')
            roc_auc_plot_path=os.path.join(plot_path, algorithm_name+'ROC_AUC.png')
            precision_recall_plot_path=os.path.join(plot_path, algorithm_name+'prec_recall.png')
        else:
            metric_plot_path=None
            roc_auc_plot_path=None
            precision_recall_plot_path=None

        if metrics_path is not None:
            average_metrics_path=os.path.join(metrics_path, algorithm_name+ '_average.xlsx')
            metrics_df_path=os.path.join(metrics_path, algorithm_name+ '.xlsx')
            # save metrics
            self.metrics_average_df.to_excel(average_metrics_path, index=False)
            self.metrics_df.to_excel(metrics_df_path, index=False)
            self.save_times(metrics_path, self.algorithm_name)

        self.make_metric_boxplot(metric_name, metric_plot_path,
                                 True, '_'.join([algorithm_name, metric_name]))

        try: #at least nearest centroid doesn't give enough metrics to make roc auc, so fail silently
            self.make_roc_auc_plot(savefile=roc_auc_plot_path,
                                                         category_index=roc_auc_plot_cat_index,
                                                         average=roc_auc_average, title=algorithm_name)
            self.make_precision_recall_plot(savefile=precision_recall_plot_path,
                                                                  category_index=roc_auc_plot_cat_index,
                                                                  average=roc_auc_average, title=algorithm_name)
        except:
            print("Failed to generate roc_auc/precision_recall plot")
            pass


    def make_roc_auc_plot(self, savefile=None, category_index=0, average=False, title=""):
        """
        make and save ROC AUC plot
         - INPUT:
            - savefile: string filename and path where to save
            - category_index: int, index of category for which information is about to be retreieved
            - average: boolean: instead of category, plot whole model macro average?
            - title: string
        - OUTPUT:
            - plot which is saved to path savefile
            """

        tprs = []
        mean_fpr = np.linspace(0, 1, 100)

        if average:
            fpr = dict()
            tpr = dict()
            roc_auc=dict()
            plt.figure()

            for j in range(self.kf.n_splits):
                # First aggregate all false positive rates
                all_fpr = np.unique(np.concatenate([self.fpr[i][j] for i in range(len(self.labels_unique))]))

                # Then interpolate all ROC curves at this points
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(len(self.labels_unique)):
                    mean_tpr += interp(all_fpr, self.fpr[i][j], self.tpr[i][j])

                # Finally average it and compute AUC
                mean_tpr /= len(self.labels_unique)

                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

                #needed for average calculation on plot
                tprs.append(interp(mean_fpr, all_fpr, mean_tpr))
                tprs[-1][0] = 0.0

                # Plot all ROC curves
                plt.plot(fpr["macro"], tpr["macro"],
                         label=r'ROC fold %d (AUC = %0.2f)' % (j, roc_auc["macro"]),
                         lw=1, alpha=0.3)

            std_auc = np.std(roc_auc['macro'])
            plt_title_category='macro'
        else: #for some specific category
            for i in range(self.kf.n_splits):
                tprs.append(interp(mean_fpr, self.fpr[category_index][i], self.tpr[category_index][i]))
                tprs[-1][0] = 0.0
                plt.plot(self.fpr[category_index][i], self.tpr[category_index][i], lw=1, alpha=0.3,
                         label='ROC fold %d (AUC = %0.2f)' % (i, self.roc_auc[category_index][i]))
                i += 1

            plt.title("ROC AUC category:" + self.labels_unique[category_index] + "_" + title)
            std_auc = np.std(self.roc_auc[category_index])
            plt_title_category=self.labels_unique[category_index]

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='b',
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
        plt.title("ROC AUC:" +plt_title_category+ "_" + title)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',  label='Luck', alpha=.8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")

        if savefile is not None:
            plt.savefig(savefile, bbox_inches='tight')

        plt.show()


    def make_precision_recall_plot(self, average=True, category_index=None, title='', savefile=None):
        """make precision recall plot based on cv results
        - INPUT:
            -average: boolean, show whole model average precision recall plot or some label,
            if False label index must be set
            - category_index: integer, index of label which precision recall plot is to be showed
            - title: string, title for plot
            - savefile: string, path+name of the file to be saved. If None it is not saved
        - OUTPUT:
            - plot and if savefile, saved to path
             """
        plt.figure()

        if average:
            title_category=category_index="micro"
        else: #some category specific
            title_category = self.labels_unique[category_index]

        for i in range(len(self.recall[0])):
            plt.step(self.recall[category_index][i], self.precision[category_index][i],
                     lw=1, alpha=0.3, where='post',
                     label='Fold %d AUC = %0.2f' % (i,
                                                    auc(self.recall[category_index][i],
                                                        self.precision[category_index][i])))
        y_real = np.concatenate(self.y_real[category_index])
        y_proba = np.concatenate(self.y_proba[category_index])
        avg_precision, avg_recall, _ = precision_recall_curve(y_real, y_proba)
        lab = 'Overall AUC=%.4f' % (auc(avg_recall, avg_precision))
        plt.step(avg_recall, avg_precision, label=lab, lw=2, color='black')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.legend(loc="lower right")
        plt.title('Precision-recall plot: '+title_category+"_"+title)

        if savefile is not None:
            plt.savefig(savefile)

        plt.show()


    def make_average_auc_boxplot(self, title=None,savefile=None):
        """make boxplot of each cv
        -INPUT:
            - title: string, title of the string
            - savefile: string, path to plot file
        - OUTPUT:
            - plot (saved if savefile is not None)"""
        fpr = dict()
        tpr = dict()
        roc_aucs=[]
        plt.figure()

        for j in range(self.kf.n_splits):
             # First aggregate all false positive rates
             all_fpr = np.unique(np.concatenate([self.fpr[i][j] for i in range(len(self.labels_unique))]))

             # Then interpolate all ROC curves at this points
             mean_tpr = np.zeros_like(all_fpr)
             for i in range(len(self.labels_unique)):
                 mean_tpr += interp(all_fpr, self.fpr[i][j], self.tpr[i][j])

             # Finally average it and compute AUC
             mean_tpr /= len(self.labels_unique)

             fpr["macro"] = all_fpr
             tpr["macro"] = mean_tpr
             roc_aucs.append( auc(fpr["macro"], tpr["macro"]))

        plt.boxplot(roc_aucs)

        axes = plt.gca()
        axes.set_ylim([0, 1])

        if title is not None:
            plt.title(title)

        if savefile is not None:
            plt.savefig(savefile, bbox_inches='tight')

        plt.show()


    def make_confusion_matrix(self,use_evaluation_data=False):
        """makes confusion matrix based on evaluation dataset
        -INPUT:
            -use_evaluation_data: boolean, if True use evaluation data instead of training data
        -OUTPUT:
            - confusion matrix
        """
        if use_evaluation_data:
            y_real = self.labels_eval_real
            y_pred = self.labels_eval_predicted
        else:
            labels_real=self.cv_labels_real
            labels_predicted=self.cv_labels_predicted
            y_real = [item for sublist in labels_real for item in sublist]
            y_pred = [item for sublist in labels_predicted for item in sublist]
        cm = confusion_matrix(y_real, y_pred, labels=self.labels_unique)
        return cm


    def plot_confusion_matrix(self,cm=None, classes=None, normalize=False, title='Confusion matrix',
                              cmap=plt.cm.Blues, use_evaluation_data=False):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        -INPUT:
            -cm: matrix, confusion matrix
            -classes:  list, list of classes
            -normalize: boolean, normalize by dividing cm col sums
            -title: str, titel of plot
            -cmap: cmap, colormap for plot
            -use_evaluation_data: boolean, if true use evaluation data
        -OUTPUT:
            -plot of confusion matrix
        """
        if classes is None:
            classes=self.labels_unique
        if cm is None:
            cm= self.make_confusion_matrix(use_evaluation_data=use_evaluation_data)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


    def predict_evaluation_set(self,texts_eval, labels_eval_real):
        """predict labels of evaluation set
        -INPUT:
            -texts_eval: list, list of texts to be used for evaluation
            -labels_eval_real: list, list of labels for texts_eval
        -OUTPUT:
            -initlize data for evaluation"""
        self.labels_eval_real=labels_eval_real
        self.labels_eval_predicted=self.text_clf.predict(list(texts_eval))


    def calc_evaluation_report(self, texts_eval,labels_eval_real, savefile=None):
        """return evaluation metrics
        -INPUT:
            -texts_eval: list, list of texts to be used for evaluation
            - labels_eval_real: list of labels to be used for evaluation
            -savefile: str, path for saving metrics files
        -OUTPUT:
            - """
        self.predict_evaluation_set(texts_eval, labels_eval_real)

        if savefile is not None:
            eval_prec_rec_f1=precision_recall_fscore_support(self.labels_eval_real, self.labels_eval_predicted,
                                                             labels=self.labels_unique)
            df_eval_metrics= pd.DataFrame(np.vstack(eval_prec_rec_f1),index=['precision', 'recall', 'f1', 'support'],
                                          columns=self.labels_unique)

            eval_prec_rec_f1_average = precision_recall_fscore_support(self.labels_eval_real, self.labels_eval_predicted,
                                                                       average="weighted", labels=self.labels_unique)
            df_eval_metrics_average = pd.DataFrame(np.vstack(eval_prec_rec_f1_average),
                                                   index=['precision', 'recall', 'f1', 'support'],
                                                   columns=['weighted'])
            df_eval_metrics.to_csv(savefile+"_"+self.algorithm_name+".csv", index=False)
            df_eval_metrics_average.to_csv(savefile+"_"+self.algorithm_name+"_average.csv", index=False)
        # return precision_recall_fscore_support(self.labels_eval_real, self.labels_eval_predicted,average=average)
        return classification_report(self.labels_eval_real, self.labels_eval_predicted)


    def pickle(self, filename):
        """save class instance to file
        -INPUT:
            -filename: str, filename to save ClassifierCv object
        -OUTPUT:
            -pickled ClassifierCv object
            """
        f = open(filename, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()


    @staticmethod
    def unpickle(filename):
        """read class instance from file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)
