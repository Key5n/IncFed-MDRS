import os
from typing import Dict
from experiments.utils.diagram.plot import plot
import numpy as np
from experiments.utils.diagram.boxplot import boxplot
import numpy as np


def save_scores(evaluation_results: list[Dict], result_dir: str):
    anomaly_scores_list = []
    labels_list = []
    auc_roc_scores = []
    auc_pr_scores = []
    vus_roc_scores = []
    vus_pr_scores = []
    pate_scores = []

    for evaluation_result in evaluation_results:
        auc_roc = evaluation_result["AUC-ROC"]
        auc_pr = evaluation_result["AUC-PR"]
        vus_roc = evaluation_result["VUS-ROC"]
        vus_pr = evaluation_result["VUS-PR"]
        pate = evaluation_result["PATE"]

        anomaly_score = evaluation_result["anomaly_score"]
        label = evaluation_result["label"]

        anomaly_scores_list.append(anomaly_score)
        labels_list.append(label)

        auc_roc_scores.append(auc_roc)
        auc_pr_scores.append(auc_pr)
        vus_roc_scores.append(vus_roc)
        vus_pr_scores.append(vus_pr)
        pate_scores.append(pate)

    auc_roc_save_path = os.path.join(result_dir, "auc_roc.csv")
    auc_pr_save_path = os.path.join(result_dir, "auc_pr.csv")
    vus_roc_save_path = os.path.join(result_dir, "vus_roc.csv")
    vus_pr_save_path = os.path.join(result_dir, "vus_pr.csv")
    pate_save_path = os.path.join(result_dir, "pate.csv")

    np.savetxt(auc_roc_save_path, auc_roc_scores)
    np.savetxt(auc_pr_save_path, auc_pr_scores)
    np.savetxt(vus_roc_save_path, vus_roc_scores)
    np.savetxt(vus_pr_save_path, vus_pr_scores)
    np.savetxt(pate_save_path, pate_scores)

    for i, (anomaly_score, label) in enumerate(zip(anomaly_scores_list, labels_list)):
        plot(anomaly_score, label, os.path.join(result_dir, f"{i}.pdf"))

        with open(os.path.join(result_dir, f"anomaly-{i}.npy"), "wb") as f:
            np.save(f, anomaly_score)

        with open(os.path.join(result_dir, f"label-{i}.npy"), "wb") as f:
            np.save(f, label)

    tick_labels = ["PATE", "VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC"]
    colors = ["red", "orange", "yellow", "green", "cyan"]
    X = [pate_scores, vus_pr_scores, vus_roc_scores, auc_pr_scores, auc_roc_scores]

    boxplot(X, tick_labels, colors, result_dir)
