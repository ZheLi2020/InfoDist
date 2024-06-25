import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support
from sklearn.metrics import auc as calc_auc

# from torchmetrics import MetricCollection
# from torchmetrics.classification import MulticlassAUROC, MulticlassRecall, MulticlassPrecision, MulticlassF1Score,
# from torcheval.metrics import MulticlassF1Score
# from torcheval.metrics.functional import multiclass_auroc, multiclass_f1_score, multiclass_precision
# from torcheval.metrics.functional.classification import multiclass_recall


def calculate_metrics(pred_list, label_list, num_classes):
    pred_list = torch.cat(pred_list, dim=0)
    label_list = torch.cat(label_list, dim=0)
    pred_prob = pred_list.softmax(dim=-1)

    auc_score = multiclass_auroc(pred_prob, label_list, num_classes=num_classes)

    fscore = multiclass_f1_score(pred_prob, label_list, num_classes=num_classes)
    precision = multiclass_precision(pred_prob, label_list)
    recall = multiclass_recall(pred_prob, label_list)

    return auc_score, fscore


def calculate_auc_and_fscore(label_pred_list, prob_list, label_true_list, num_classes):
    label_true = np.concatenate(label_true_list, axis=0)
    label_pred = np.concatenate(label_pred_list, axis=0)
    prob_pred = np.concatenate(prob_list, axis=0)

    auc_score = 0

    if num_classes > 2:
        aucs = []
        binary_labels = label_binarize(label_true, classes=[i for i in range(num_classes)])
        for class_idx in range(num_classes):
            if class_idx in label_true:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob_pred[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc_score = np.nanmean(np.array(aucs))
    else:
        auc_score = roc_auc_score(label_true, prob_pred)

    precision, recall, fscore, _ = precision_recall_fscore_support(label_true, label_pred, zero_division=0)

    # precision_mean = np.mean(precision)
    # recall_mean = np.mean(recall)
    fscore_mean = np.mean(fscore)

    return auc_score, fscore_mean



