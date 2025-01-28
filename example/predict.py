import sys
sys.path.append('../')
import json
import logging
import argparse
from deeplog.deeplog import model_fn, input_fn, predict_fn
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
import numpy as np

logging.basicConfig(level=logging.WARNING,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=int, default=23, metavar='N',
                        help='to determine the time series data is an anomaly or not.')
    args = parser.parse_args()

    ##############
    # Load Model #
    ##############
    model_dir = './model'
    model_info = model_fn(model_dir)

    ###########
    # predict #
    ###########
    test_abnormal_list = []
    with open('test_abnormal', 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n - 1, map(int, line.strip().split())))
            request = json.dumps({'line': line})
            input_data = input_fn(request, 'application/json')
            response = predict_fn(input_data, model_info)
            test_abnormal_list.append(response)

    test_normal_list = []
    with open('test_normal', 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n - 1, map(int, line.strip().split())))
            request = json.dumps({'line': line})
            input_data = input_fn(request, 'application/json')
            response = predict_fn(input_data, model_info)
            test_normal_list.append(response)

    ##############
    # Evaluation #
    ##############
    thres = args.threshold
    abnormal_has_anomaly = [1 if t['anomaly_cnt'] > thres else 0 for t in test_abnormal_list]
    abnormal_cnt_anomaly = [t['anomaly_cnt'] for t in test_abnormal_list]
    abnormal_predict = []
    for test_abnormal in test_abnormal_list:
        abnormal_predict += test_abnormal['predict_list']

    normal_has_anomaly = [1 if t['anomaly_cnt'] > thres else 0 for t in test_normal_list]
    normal_cnt_anomaly = [t['anomaly_cnt'] for t in test_normal_list]
    normal_predict = []
    for test_normal in test_normal_list:
        normal_predict += test_normal['predict_list']

    ground_truth = [1]*len(abnormal_has_anomaly) + [0]*len(normal_has_anomaly)
    predict = abnormal_has_anomaly + normal_has_anomaly
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    accu = 0
    for p, t in zip(predict, ground_truth):
        if p == t:
            accu += 1

        if p == 1 and t == 1:
            TP += 1
        elif p == 1 and t == 0:
            FP += 1
        elif p == 0 and t == 1:
            FN += 1
        else:
            TN += 1

    logger.info(f'thres: {thres}')
    logger.info(f'TP: {TP}')
    logger.info(f'FP: {FP}')
    logger.info(f'TN: {TN}')
    logger.info(f'FN: {FN}')

    accuracy = accu / len(predict)
    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    logger.info(f'accuracy: {accuracy}')
    logger.info(f'Precision: {precision}')
    logger.info(f'Recall: {recall}')
    logger.info(f'F1: {F1}')

    #-----------------Visualization-------------------------------------#
    
    # Create a display for the confusion matrix
    conf_matrix = confusion_matrix(ground_truth, predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Normal", "Abnormal"])
    # Plot the confusion matrix
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    # Save the confusion matrix plot
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix plot saved as confusion_matrix.png")

    # Generate ROC curve values
    fpr, tpr, _ = roc_curve(ground_truth, predict)
    roc_auc = auc(fpr, tpr)
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    # Save the ROC curve
    plt.savefig("roc_curve.png")
    print("ROC curve saved as roc_curve.png")


    # Threshold values to evaluate
    thresholds = np.arange(0, max(max(abnormal_cnt_anomaly), max(normal_cnt_anomaly)) + 1, 1)

    # Initialize lists for metrics
    precision_vals = []
    recall_vals = []
    f1_vals = []

    # Calculate metrics for each threshold
    for thres in thresholds:
        preds = [1 if cnt > thres else 0 for cnt in abnormal_cnt_anomaly + normal_cnt_anomaly]
        tp = sum(1 for p, t in zip(preds, ground_truth) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(preds, ground_truth) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(preds, ground_truth) if p == 0 and t == 1)

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        precision_vals.append(precision)
        recall_vals.append(recall)
        f1_vals.append(f1)

    # Plot Precision, Recall, and F1 vs. Threshold
    plt.figure()
    plt.plot(thresholds, precision_vals, label="Precision", lw=2)
    plt.plot(thresholds, recall_vals, label="Recall", lw=2)
    plt.plot(thresholds, f1_vals, label="F1 Score", lw=2)
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.title("Metrics vs. Threshold")
    plt.legend(loc="best")

    # Save the Metrics vs. Threshold plot
    plt.savefig("metrics_vs_threshold.png")
    print("Metrics vs. Threshold plot saved as metrics_vs_threshold.png")
