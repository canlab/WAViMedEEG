import matplotlib.pyplot as plt
import config
import numpy as np


def roc(y_preds, y_labels, fname=None, plot=True):
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

    if plot is True:
        fig1 = plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(
            fpr,
            tpr,
            label='(area = {:.3f})'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='best')

    if fname is None:
        plt.show()

    if fname is not None:
        fig1.savefig(fname)
        plt.close(fig1)

    return auc_keras


def plot_signal(t, sig):
    plt.figure(1)
    plt.plot(t, sig)
    plt.ylim(np.min(sig) - 1, np.max(sig) + 1)
    plt.show()


def plot_svm_features(
    Features,
    svm_weights,
    scores=None,
    network_channels=config.network_channels,
        fname=None):

    # set up figure and axes (rows)
    fig, axs = plt.subplots(
        nrows=len(network_channels),
        figsize=(20, 40))

    plt.rcParams['figure.dpi'] = 200

    # plt.clf()

    # X_indices = np.arange(train_dataset.shape[-1])
    X_indices = Features

    i = 0
    j = 0
    for channel in network_channels:

        axs[j].set_title(channel)
        axs[j].bar(
                X_indices - .25,
                svm_weights[i:i + len(X_indices)],
                width=.1, label='SVM weight')

        if scores is None:
            axs[j].legend()

        i += len(X_indices)
        j += 1

    if scores is not None:

        i = 0
        j = 0
        for channel in network_channels:
            axs[j].bar(
                X_indices - .45,
                scores[i:i+len(X_indices)],
                width=.1,
                label=r'Univariate score ($-Log(p_{value})$)')

            axs[j].legend()

            i += len(X_indices)
            j += 1

    axs[0].set_title("Comparing feature selection")
    axs[-1].set_xlabel('Feature number (in Hz)')
    # plt.yticks(())
    fig.tight_layout()
    plt.show()

    if fname is not None:
        fig.savefig(fname)
        plt.close(fig)
