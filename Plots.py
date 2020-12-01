import matplotlib.pyplot as plt

def roc(y_preds, y_labels, fname="ROC", plot=True):
    """
    ROC curve plotting function

    Parameters:
        - y_preds: (list) floats
            predictions for y_data
        - y_labels: (list) strings
            true labels for y_data
        - fname: (str) default 'ROC'
            plot filename to save as
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(
        y_labels,
        y_preds,
        pos_label=1)

    from sklearn.metrics import auc
    auc_keras = auc(fpr, tpr)

    if plot==True:
        fig1 = plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr,
            tpr,
            label='(area = {:.3f})'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='best')
        plt.show()
        fig1.savefig(fname)

    return auc_keras

def plot_signal(t, sig):
    plt.figure(1)
    plt.plot(t, sig)
    plt.ylim(np.min(sig) - 1, np.max(sig) + 1)
    plt.show()
