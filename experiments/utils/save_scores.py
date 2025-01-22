import os
from typing import Dict
from experiments.utils.diagram.plot import plot
import numpy as np
from experiments.utils.diagram.boxplot import boxplot
import numpy as np


def save_scores(result: Dict, result_dir: str):
    anomaly_scores = result["anomaly_scores"]
    label_list = result["labels_list"]
    auc_roc_scores = result["auc_roc_scores"]
    auc_pr_scores = result["auc_pr_scores"]
    vus_roc_scores = result["vus_roc_scores"]
    vus_pr_scores = result["vus_pr_scores"]
    pate_scores = result["pate_scores"]

    pate_save_path = os.path.join(result_dir, "pate.csv")
    vus_pr_save_path = os.path.join(result_dir, "vus_pr.csv")
    vus_roc_save_path = os.path.join(result_dir, "vus_roc.csv")
    auc_pr_save_path = os.path.join(result_dir, "auc_pr.csv")
    auc_roc_save_path = os.path.join(result_dir, "auc_roc.csv")

    np.savetxt(pate_save_path, pate_scores)
    np.savetxt(vus_pr_save_path, vus_pr_scores)
    np.savetxt(vus_roc_save_path, vus_roc_scores)
    np.savetxt(auc_pr_save_path, auc_pr_scores)
    np.savetxt(auc_roc_save_path, auc_roc_scores)

    for i, (anomaly_score, label) in enumerate(zip(anomaly_scores, label_list)):
        plot(anomaly_score, label, os.path.join(result_dir, f"{i}.pdf"))

    tick_labels = ["PATE", "VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC"]
    colors = ["red", "orange", "yellow", "green", "cyan"]
    X = [pate_scores, vus_pr_scores, vus_roc_scores, auc_pr_scores, auc_roc_scores]

    boxplot(X, tick_labels, colors, result_dir)
