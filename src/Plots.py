from src import config
import matplotlib.pyplot as plt
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
        y_preds)

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


def pred_hist(
    y_preds,
    y_labels,
    fname=None,
    plot=True,
    group_names=['', '']):
    """
    Prediction Histogram

    Parameters:
        - y_preds: (list) floats
            predictions for y_data
        - y_labels: (list) strings
            true labels for y_data
        - fname: (str) default 'ROC'
            plot filename to save as
    """

    correct = 0
    for pred, true in zip(y_preds, y_labels):
        if np.rint(pred) == True:
            correct += 1
    acc = correct / len(y_preds)

    if plot is True:
        fig1 = plt.figure(1)
        plt.hist(
            y_preds,
            label='(acc = {:.3f})'.format(acc),
            bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.xlabel(
            "Pr("\
            + group_names[0]\
            + ') = 1'\
            + ' -> '\
            + 'Pr('\
            + group_names[1]\
            + ") = 1")
        plt.ylabel('Count')
        plt.title('Model Evaluation')
        plt.legend(loc='best')

    if fname is None:
        plt.show()

    if fname is not None:
        fig1.savefig(fname)
        plt.close(fig1)

    return acc


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


def plot_LDA(lda, X, y, y_pred):
    from matplotlib import colors
    # Colormap
    cmap = colors.LinearSegmentedColormap(
        'red_blue_classes',
        {'red': [(0, 1, 1), (1, 0.7, 0.7)],
         'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
         'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
    plt.cm.register_cmap(cmap=cmap)

    # Plot functions
    def plot_data(lda, X, y, y_pred):
        fig = plt.plot()
        plt.title('Linear Discriminant Analysis')

        tp = (y == y_pred)  # True Positive
        tp0, tp1 = tp[y == 0], tp[y == 1]
        X0, X1 = X[y == 0], X[y == 1]
        X0_tp, X0_fp = X0[tp0], X0[~tp0]
        X1_tp, X1_fp = X1[tp1], X1[~tp1]

        # class 0: dots
        plt.scatter(
            X0_tp[:, 0],
            X0_tp[:, 1],
            marker='.',
            color='red')

        plt.scatter(
            X0_fp[:, 0],
            X0_fp[:, 1],
            marker='x',
            s=20,
            color='#990000')  # dark red

        # class 1: dots
        plt.scatter(
            X1_tp[:, 0],
            X1_tp[:, 1],
            marker='.',
            color='blue')

        plt.scatter(
            X1_fp[:, 0],
            X1_fp[:, 1],
            marker='x',
            s=20,
            color='#000099')  # dark blue

        # class 0 and 1 : areas
        nx, ny = 200, 100
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, nx),
            np.linspace(y_min, y_max, ny))

        Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 1].reshape(xx.shape)
        plt.pcolormesh(
            xx,
            yy,
            Z,
            cmap='red_blue_classes',
            norm=colors.Normalize(0., 1.),
            zorder=0)

        plt.contour(
            xx,
            yy,
            Z,
            [0.5],
            linewidths=2.,
            colors='white')

        # means
        plt.plot(
            lda.means_[0][0],
            lda.means_[0][1],
            '*',
            color='yellow',
            markersize=15,
            markeredgecolor='grey')

        plt.plot(
            lda.means_[1][0],
            lda.means_[1][1],
            '*',
            color='yellow',
            markersize=15,
            markeredgecolor='grey')

        return splot

    def plot_ellipse(splot, mean, cov, color):
        v, w = linalg.eigh(cov)
        u = w[0] / linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        # filled Gaussian at 2 standard deviation
        ell = mpl.patches.Ellipse(
            mean,
            2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
            180 + angle,
            facecolor=color,
            edgecolor='black',
            linewidth=2)

        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.2)
        splot.add_artist(ell)
        splot.set_xticks(())
        splot.set_yticks(())

    def plot_lda_cov(lda, splot):
        plot_ellipse(splot, lda.means_[0], lda.covariance_, 'red')
        plot_ellipse(splot, lda.means_[1], lda.covariance_, 'blue')

    splot = plot_data(lda, X, y, y_pred)
    plot_lda_cov(lda, splot)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()


def plot_layer_size_covariance(sizes=[], values=[], metric=""):
    for dataset in values:
        plt.plot(sizes, dataset)
    plt.legend(labels=['Training', 'Validation'])
    plt.title("Performance Metric vs. Depth")
    plt.xlabel("# of Convolutional Layers (Depth)")
    plt.ylabel("Performance Metric Value")
    if metric != "":
        plt.ylabel(metric)
    plt.show()


def plot_confusion_matrix(cm, checkpoint_dir, class_names, fname=""):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    import itertools
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(
        cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],
        decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(checkpoint_dir+"/confusion_matrix"+fname)
    plt.clf()

    return figure


def plot_history(clf, history, checkpoint_dir, metric):
    # save accuracy curve to log dir
    plt.plot(
     history.history[metric],
     label='train: '\
         + "Subs: " + str(len(clf.train_subjects)) + " "\
         + "Data: " + str(len(clf.train_dataset)))

    plt.plot(
     history.history['val_'+metric],
     label='test: '\
         + "Subs: " + str(len(clf.test_subjects)) + " "\
         + "Data: " + str(len(clf.test_dataset)))
    plt.title('Epoch ' + str(metric))
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig(checkpoint_dir+"/epoch_"+metric)
    plt.clf()


def plot_3d_scatter(y_preds, y_labels, label_names, checkpoint_dir, fig_fname):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import animation
    def rotate(angle):
        ax.view_init(azim=angle)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for class_label in list(set(y_labels)):
        preds = []
        for pred, label in zip(y_preds, y_labels):
            if label == class_label:
                preds.append(pred)
        ax.scatter(
            [pred[0] for pred in preds],
            [pred[1] for pred in preds],
            [pred[2] for pred in preds],
            label=label_names[class_label])

    ax.set_xlabel(label_names[0])
    ax.set_ylabel(label_names[1])
    ax.set_zlabel(label_names[2])
    plt.legend()
    plt.grid(True)

    rot_animation = animation.FuncAnimation(
        fig,
        rotate,
        frames=np.arange(0, 362, 2),
        interval=100)
    rot_animation.save(
        checkpoint_dir+"/"+fig_fname+".gif",
        dpi=100,
        writer='imagemagick')

    # plt.savefig(checkpoint_dir+"/"+fig_fname)
    plt.clf()
    plt.close()
