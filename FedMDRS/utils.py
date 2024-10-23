import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report, precision_score, recall_score, auc, roc_curve

def generate_graph(test_label, threshold, mahalanobis_distances, filename):
    plt.clf()
    x = np.arange(len(test_label))
    plt.axhline(y=threshold, color="green", linestyle="--", label="threshold")
    plt.plot(x, mahalanobis_distances, label="Mahalanobis Distance")

    for i in range(len(test_label)):
        if test_label[i] == 1:
            plt.axvspan(i, i, color="red", alpha=0.3)

    plt.legend()
    plt.savefig(filename)

def write_analysis(dirname, label_test, label_pred):
    with open(f"result/{dirname}/system.txt", "w") as o:
        print("confusion_matrix", confusion_matrix(label_test, label_pred), file=o)
        print("precision score", precision_score(label_test, label_pred), file=o)
        print("recall score", recall_score(label_test, label_pred), file=o)
        print("f1 score", f1_score(label_test, label_pred),file=o)
        print(classification_report(label_test, label_pred), file=o)

def write_roc_curve(false_positives, true_positives, roc_auc, filename):
    plt.clf()
   # Plot ROC Curve
    plt.plot(false_positives, true_positives, marker='o', label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(filename)

