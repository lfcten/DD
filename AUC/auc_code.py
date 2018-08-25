# -*- coding: utf-8 -*-
import random

from sklearn import metrics

"""
1. 近似计算曲线下面积
"""


def roc_auc_score_1(y_true, y_score):
    pos = sum(y_true)
    neg = len(y_true) - pos
    data = sorted(list(zip(y_true, y_score)), key=lambda s: s[1], reverse=True)
    res = []
    tp, fp = 0, 0
    for label, score in data:
        if label == 1:
            tp += 1
        else:
            fp += 1
        res.append([fp / neg, tp / pos])

    auc = 0.
    prev_fpr = 0
    for fpr, tpr in res:
        if fpr != prev_fpr:
            auc += (fpr - prev_fpr) * tpr
            prev_fpr = fpr
    return auc


"""
2. 公式计算
"""


def roc_auc_score_2(y_true, y_score):
    n = len(y_true)
    assert n >= 1
    pos = sum(y_true)
    neg = n - pos
    data = sorted(list(zip(y_true, y_score)), key=lambda s: s[1])
    auc = 0
    prev_score = data[0][1]
    rank_sum, pos_count = 0, 0
    if data[0][0]:
        rank_sum = 1
        pos_count = 1
    count = 1
    for i in range(1, n):
        if data[i][1] != prev_score:
            auc += pos_count / count * rank_sum
            pos_count = 0
            count = 0
            rank_sum = 0
            prev_score = data[i][1]

        if data[i][0] == 1:
            pos_count += 1
        count += 1
        rank_sum += i + 1
    auc += pos_count / count * rank_sum
    auc = (auc - pos * (pos + 1) / 2) / (pos * neg)
    return auc


if __name__ == "__main__":
    y_true, y_score = [random.randint(0, 1) for _ in range(100)], [random.random() for _ in range(100)]
    auc1 = roc_auc_score_1(y_true, y_score)
    auc2 = roc_auc_score_2(y_true, y_score)
    auc_standard = metrics.roc_auc_score(y_true, y_score)

    print("auc_1: %s   auc_2: %s  auc_standard: %s" % (auc1, auc2, auc_standard))
