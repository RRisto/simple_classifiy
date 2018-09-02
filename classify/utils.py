import numpy as np
from matplotlib import pyplot as plt, cm
from sklearn.metrics import auc
from scipy import interp
import itertools


def make_comparison_roc_auc_plot(classifier_instances, savefile=None, category_index=0, average=False, title=""):
    """
    make and save comparsion of ROC AUC plots
     - INPUT:
        - classifier_instances: list of instance of ClassifierCv, which are trained and which ROC AUC comparison
         plot to be made
        - savefile: string filename and path where to save
        - category_index: int, index of category for which information is about to be retreieved
        - average: boolean: instead of category, plot whole model macro average?
        - title: string
    - OUTPUT:
        - plot which is saved to path savefile
        """

    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure()
    plt_title_category = ''

    for classifier in classifier_instances:
        if average:
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for j in range(classifier.kf.n_splits):
                # First aggregate all false positive rates
                all_fpr = np.unique(
                    np.concatenate([classifier.fpr[i][j] for i in range(len(classifier.labels_unique))]))

                # Then interpolate all ROC curves at this points
                mean_tpr = np.zeros_like(all_fpr)
                for i in range(len(classifier.labels_unique)):
                    mean_tpr += interp(all_fpr, classifier.fpr[i][j], classifier.tpr[i][j])

                # Finally average it and compute AUC
                mean_tpr /= len(classifier.labels_unique)

                fpr["macro"] = all_fpr
                tpr["macro"] = mean_tpr
                roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

                # needed for average calculation on plot
                tprs.append(interp(mean_fpr, all_fpr, mean_tpr))
                tprs[-1][0] = 0.0

            std_auc = np.std(roc_auc['macro'])
            plt_title_category = 'macro'
        else:  # for some specific category
            for i in range(classifier.kf.n_splits):
                tprs.append(interp(mean_fpr, classifier.fpr[category_index][i], classifier.tpr[category_index][i]))
                tprs[-1][0] = 0.0
                i += 1

            std_auc = np.std(classifier.roc_auc[category_index])
            plt_title_category = classifier.labels_unique[category_index]

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr,  # color=next(colors),
                 label=r'%s mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (str(classifier.text_clf._final_estimator),
                                                                   mean_auc, std_auc), lw=2, alpha=.8)

    plt.title("ROC AUC:" + plt_title_category + "_" + title)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    lgd = plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2))

    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight')

    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, savefile=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

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
    plt.ylabel('Tegelik kategooria')
    plt.xlabel('Ennustatud kategooria')
    if savefile is not None:
        plt.tight_layout()
        plt.savefig(savefile)
