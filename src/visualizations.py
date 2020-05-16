import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics


def evaluate_text(y_true, y_pred, y_proba):
    print(f'accuracy = {metrics.accuracy_score(y_true, y_pred):.3f}')
    print(f'precision = {metrics.precision_score(y_true, y_pred):.3f}')
    print(f'recall = {metrics.recall_score(y_true, y_pred):.3f}')
    auc_score = metrics.roc_auc_score(y_true,
                                      y_proba[:, 1], average='weighted')
    print(f'AUC = {auc_score:.3f}')


def visualize_confusion_matrix(y_true, y_predict):
    # Construct the confusion matrix using the predicted and actual labels
    cm = metrics.confusion_matrix(y_true, y_predict)
    C_df = pd.DataFrame(cm,
                        index=['Live', 'Die'],
                        columns=['Live', 'Die'])

    _fig, ax = plt.subplots(figsize=(2.5, 2.5))
    sns.heatmap(C_df,
                cmap='Blues', cbar=False,
                annot=True, fmt='d', annot_kws={"fontsize": 12}, )
    ax.tick_params(labelsize=14, which='both')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')


def plot_intersection(x, y, ax):
    '''
        plot the intersection lines for an x, y coordinate and annote it
    '''
    ax.plot([0, x], [y, y], 'r:')
    ax.plot([x, x], [0, y], 'r:')
    ax.plot([x], [y], 'ro')
    ax.annotate(xy=(x + 0.01, y), s=f'({x:.2f}, {y:.2f})', c='r')


def plot_precision_vs_recall(precisions, recalls, precision, recall, ax):
    '''
    adapted from:
    Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn,
        Keras, and TensorFlow, 2nd Edition. O'Reilly Media, Inc.
    '''
    ax.plot(recalls, precisions, "b-", linewidth=2)
    ax.set_xlabel("Recall", fontsize=16)
    ax.set_ylabel("Precision", fontsize=16)
    ax.set_title("Precision vs Recall Curve", size=20)
    ax.tick_params(labelsize=14)
    ax.axis([0, 1.01, 0, 1.01])
    ax.grid(True)
    plot_intersection(recall, precision, ax)


def plot_roc_curve(fprs, tprs, fpr, tpr, auc_score, ax, label=None):
    '''
    adapted from:
    Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn,
        Keras, and TensorFlow, 2nd Edition. O'Reilly Media, Inc.
    '''
    ax.plot(fprs, tprs, "b-", linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate (1 - Specivity)', fontsize=16)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=16)
    ax.set_title("ROC Curve", size=20)
    ax.tick_params(labelsize=14)
    ax.axis([-0.01, 1.01, 0, 1.01])
    ax.grid(True)
    auc = f'AUC = {auc_score:.3f}'
    ax.annotate(xy=(0.7, 0.05), s=auc, c='b')
    plot_intersection(fpr, tpr, ax)


def plot_precision_and_roc(y_true, y_pred, y_prob):
    _fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    precisions, recalls, _thresholds =  \
        metrics.precision_recall_curve(y_true, y_prob[:, 1])
    p_score = metrics.precision_score(y_true, y_pred)
    r_score = metrics.recall_score(y_true, y_pred)
    auc_score = metrics.roc_auc_score(y_true, y_pred)

    plot_precision_vs_recall(precisions, recalls, p_score, r_score, axs[0])

    fprs, tprs, _thresholds = metrics.roc_curve(y_true, y_prob[:, 1])
    _tn, _fp, _fn, _tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    fpr = _fp / (_fp + _tn)

    plot_roc_curve(fprs, tprs, fpr, r_score, auc_score, axs[1])


def evaluate_model(clf, X, y_true, description, plot=True):
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)

    print(description)
    evaluate_text(y_true, y_pred, y_prob)
    if plot:
        visualize_confusion_matrix(y_true, y_pred)
        plot_precision_and_roc(y_true, y_pred, y_prob)


if __name__ == '__main__':
    print("does nothing")
