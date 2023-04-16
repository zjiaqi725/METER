import os
import random
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,auc,precision_recall_curve


def shingle(series, dim):
    """takes a one dimensional series and shingles it into dim dimensions"""
    height = len(series) - dim + 1
    shingled = np.zeros((dim, height))
    for i in range(dim):
        shingled[i] = series[i:i + height]
    return shingled



def get_result(args, y_test, recon_err_test, n_lin=200, decimal=4, verbose=False):
    """   Compute the metrics for anomaly detection results   """
    threshold = np.linspace(min(recon_err_test),max(recon_err_test), n_lin)
    acc_list = []
    f1_list = []
    auc_list = []
    for t in threshold:
        y_pred = (recon_err_test>t).astype(np.int)
        acc_list.append(accuracy_score(y_pred,y_test))
        f1_list.append(f1_score(y_pred,y_test))
        auc_list.append(roc_auc_score(y_test,y_pred ))
    
    i = np.argmax(f1_list)   
    t = threshold[i]
    score = f1_list[i]
    print('Recommended threshold: %.3f, related f1 score: %.3f'%(t,score))
    # t = sorted(recon_err_test, reverse=True)[int((sum(y_test)/len(y_test))*len(y_test))]  #按已知异常率设定阈值
    # print('Recommended threshold: %.3f'%(t))
    y_pred = (recon_err_test>t).astype(np.int)
    print('\nTest set : AUC_ROC: {:.3f}  F1:{:.3f}  Accuracy:{:.3f}'.format(roc_auc_score(y_test,recon_err_test ),f1_score(y_pred,y_test) ,accuracy_score(y_pred,y_test))) 
    FN = ((y_test==1) & (y_pred==0)).sum()
    FP = ((y_test==0) & (y_pred==1)).sum()
    TP = ((y_test==1) & (y_pred==1)).sum()
    print('precision: {:.4f}'.format(TP/(TP+FP)))
    print('Recall: {:.4f}'.format(TP/(FN+TP)))
#    print('precision: {:.4f}'.format(precision))
#    print('Recall: {:.4f}'.format(recall))

    precision, recall, _thresholds = precision_recall_curve(y_test, recon_err_test)
    area = auc(recall, precision)
    print('AUC_PR: {:.4f}'.format(area))
    print('TP: {:.4f}'.format(TP),'FP: {:.4f}'.format(FP),'FN: {:.4f}'.format(FN))
    metrics_dict = {}
    metrics_dict['dataset'] = args.dataset
    metrics_dict['accuracy'] = round(accuracy_score(y_pred,y_test), decimal)
    metrics_dict['precision'] = round(TP/(TP+FP), decimal)
    metrics_dict['recall'] = round(TP/(FN+TP), decimal)
    metrics_dict['f_score'] = round(f1_score(y_pred,y_test), decimal)
    metrics_dict['auc_roc'] = round(roc_auc_score(y_test,recon_err_test), decimal)
    metrics_dict['auc_pr'] = round(auc(recall, precision), decimal)
    metrics_dict['minscore'] = round(min(recon_err_test), decimal)
    metrics_dict['maxscore'] = round(max(recon_err_test), decimal)
    metrics_dict['average'] = round(recon_err_test.mean(), decimal)
    if verbose:
        print(pd.DataFrame(metrics_dict, index=['Results']))
    return metrics_dict, y_pred


