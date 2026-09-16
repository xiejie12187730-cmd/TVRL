# this code is released by:
# L. Yang, X.-Z. Wu, Y. Jiang, and Z.-H. Zhou.
# Multi-label deep forest.
# In: Proceedings of the 24th European Conference on Artificial Intelligence (ECAI'20),
# Santiago de Compostela, Spain, 2020. [code]

import numpy as np
from sklearn import metrics
from sklearn.metrics import label_ranking_loss


def do_metric(y_prob, label):
    y_predict = y_prob > 0.5
    ranking_loss = 1 - label_ranking_loss(label, y_prob)
    one_error = 1 - compute_one_error(y_prob, label)
    coverage = 1 - compute_coverage(y_prob, label)
    hamming_loss = 1 - compute_hamming_loss(y_predict, label)
    precision = compute_average_precision(y_prob, label)
    macro_f1 = compute_macro_f1(y_predict, label)
    micro_f1 = compute_micro_f1(y_predict, label)
    auc = compute_auc(y_prob, label)
    auc_me = calc_auc(y_prob, label)

    return np.array([precision, hamming_loss, ranking_loss, auc_me, one_error, coverage, auc, macro_f1, micro_f1])


def compute_accuracy(pred_label, label):
    num_samples = len(label)
    acc = sum(label == pred_label) * 1.0 / num_samples
    return acc


def compute_rank(y_prob):
    rank = np.zeros(y_prob.shape)
    for i in range(len(y_prob)):
        temp = y_prob[i, :].argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(y_prob[i, :]))
        rank[i, :] = ranks
    return y_prob.shape[1] - rank


def compute_hamming_loss(pred_label, label):
    acc = compute_accuracy(pred_label, label)
    return 1 - acc.mean()


def compute_macro_f1(pred_label, label):
    up = np.sum(pred_label * label, axis=0)
    down = np.sum(pred_label, axis=0) + np.sum(label, axis=0)
    if np.sum(np.sum(label, axis=0) == 0) > 0:
        up[down == 0] = 0
        down[down == 0] = 1
    macro_f1 = 2.0 * np.sum(up / down)
    macro_f1 = macro_f1 * 1.0 / label.shape[1]
    return macro_f1


def compute_micro_f1(pred_label, label):
    up = np.sum(pred_label * label)
    down = np.sum(pred_label) + np.sum(label)
    if np.sum(np.sum(label) == 0) > 0:
        up[down == 0] = 0
        down[down == 0] = 1
    micro_f1 = 2.0 * up / down
    return micro_f1


def compute_one_error(y_prob, label):
    num_samples, _ = label.shape
    loss = 0
    for i in range(num_samples):
        pos = np.argmax(y_prob[i, :])
        loss += label[i, pos] < 0.5
    return loss * 1.0 / num_samples


def compute_coverage(y_prob, label):
    num_samples, num_labels = label.shape
    rank = compute_rank(y_prob)
    coverage = 0
    for i in range(num_samples):
        if sum(label[i, :] > 0.5) > 0:
            coverage += max(rank[i, label[i, :] > 0.5])
    coverage = coverage * 1.0 / num_samples - 1
    return coverage / num_labels


def compute_average_precision(y_prob, label):
    num_samples, _ = label.shape
    rank = compute_rank(y_prob)
    precision = 0
    for i in range(num_samples):
        positive = np.sum(label[i, :] > 0.5)
        rank_i = rank[i, label[i, :] > 0.5]
        temp = rank_i.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(rank_i))
        ranks = ranks + 1
        ans = ranks * 1.0 / rank_i
        if positive > 0:
            precision += np.sum(ans) * 1.0 / positive
    return precision / num_samples


def compute_auc(y_prob, label):
    _, num_labels = label.shape
    macro_auc = 0
    valid_labels = 0
    for i in range(num_labels):
        if np.unique(label[:, i]).shape[0] == 2:
            index = np.argsort(y_prob[:, i])
            pred = y_prob[:, i][index]
            y = label[:, i][index] + 1
            fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
            temp = metrics.auc(fpr, tpr)
            macro_auc += temp
            valid_labels += 1
    macro_auc /= valid_labels
    return macro_auc


def calc_roc(y_pred, y_true):
    sorted_indices = np.argsort(y_pred, axis=1)[:, ::-1]
    y_true_sorted = np.take_along_axis(y_true, sorted_indices, axis=1).astype(int)

    n = y_true.shape[0]
    c = y_true.shape[1]
    tpr = []
    fpr = []
    for i in range(c):
        tp = np.sum(y_true_sorted[:, :i + 1])
        fp = n * (i + 1) - tp
        fn = np.sum(y_true_sorted[:, i + 1:])
        tn = n * (c - i - 1) - fn
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
    tpr = np.array(tpr)
    fpr = np.array(fpr)

    return tpr, fpr


def calc_auc(y_pred, y_true):
    tpr, fpr = calc_roc(y_pred, y_true)

    diff_fpr = np.diff(fpr)
    avg_tpr = (tpr[:-1] + tpr[1:]) / 2
    area = np.sum(diff_fpr * avg_tpr)
    value_auc = area / (fpr[-1] - fpr[0])
    return value_auc
