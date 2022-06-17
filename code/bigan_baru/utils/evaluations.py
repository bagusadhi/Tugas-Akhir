import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, precision_recall_curve, auc
import pandas as pd
import sys


def do_prc(scores, true_labels, file_name='', directory='', plot=True):
    """ Does the PRC curve

    Args :
            scores (list): list of scores from the decision function
            true_labels (list): list of labels associated to the scores
            file_name (str): name of the PRC curve
            directory (str): directory to save the jpg file
            plot (bool): plots the PRC curve or not
    Returns:
            prc_auc (float): area under the under the PRC curve
    """
    # print(f"SCORES: \n {scores}")
    # print(f"Jumlah score : {len(scores)}")
    # y_pred = []
    # idx = 0
    # for i in range(1,int(3200/32)):
    #     num = 32 * i
    #     y_pred = np.array(scores[idx:num])
    #     idx += num

    #     y_pred = np.mean(np.log(np.maximum(1.0 - y_pred, sys.float_info.epsilon)) - np.log(np.maximum(y_pred, sys.float_info.epsilon)))
    #     # y_pred.append(y_pred)
    #     print(y_pred)

# ====================================================================================================================================================

    # np.savetxt('/TA/bagus_adhi/BiGAN/bigan_2022/scores.csv',scores,delimiter=',')
    # np.savetxt('/TA/bagus_adhi/BiGAN/bigan_2022/true_labels.csv',true_labels,delimiter=',')
    #print(f"TRUE LABELS : \n{true_labels}")
    #print(f"Jumlah true labels : {len(true_labels)}")


    # anomaly_score_csv = '/TA/bagus_adhi/BiGAN/bigan_2022/anomaly_score_ToyTrain_section_00_source_test.csv'
    # anomaly_score_list = []
    # anomaly_score_list.append([os.path.basename('/TA/bagus_adhi/dev_data/ToyTrain/test'), scores])
    # print(y_pred)



    
    # file = []

    # for i in os.listdir("/TA/bagus_adhi/dev_data/bearing/test"):
    #     filename = i.split('_')
    #     if filename[1] == '00' and filename[2] == 'target':#pilih source atau target
    #         filename = i
    #         # filename = np.array(filename)
    #         file.append(filename)

    #         # print(filename)
    # file = np.array(file)
    # scores = np.array(scores)
    # # print(file)
    # # print(scores)
    # df = pd.DataFrame({"name" : file, "anomaly_score" : scores})
    # df.to_csv('/TA/bagus_adhi/BiGAN/bigan_2022/anomaly_score_bearing_section_00_target_test.csv', index=False)

    
    precision, recall, thresholds = precision_recall_curve(true_labels, scores)
    # print(f"PRECISION : \n{precision}")
    # print(f"y_pred : {y_pred.shape}")
    # print(f"Jumlah true labels : {len(true_labels)}")
    # # y_true = true_labels[: : 32]

    # np.savetxt('/TA/bagus_adhi/BiGAN/bigan_2022/precision.csv',precision,delimiter=',')
    # np.savetxt('/TA/bagus_adhi/BiGAN/bigan_2022/recall.csv',recall,delimiter=',')
    
    prc_auc = roc_auc_score(true_labels, scores)
    p_auc = roc_auc_score(true_labels, scores, max_fpr=0.1)

    if plot:
        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curve: AUC=%0.4f' 
                            %(prc_auc) + ' pAUC=%0.4f'%(p_auc))
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig('results/' + file_name + '_prc.jpg')
        plt.close()

    return prc_auc


