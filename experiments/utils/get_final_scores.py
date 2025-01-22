import os
from logging import getLogger
from typing import Dict
import numpy as np
from experiments.utils.diagram.boxplot import boxplot
import numpy as np


def get_final_scores(evaluation_results: list[Dict], result_dir: str):
    logger = getLogger(__name__)

    pate_scores = []
    vus_pr_scores = []
    vus_roc_scores = []
    auc_pr_scores = []
    auc_roc_scores = []

    for evaluation_result in evaluation_results:
        pate = evaluation_result["PATE"]
        vus_pr = evaluation_result["VUS-PR"]
        vus_roc = evaluation_result["VUS-ROC"]
        auc_pr = evaluation_result["AUC-PR"]
        auc_roc = evaluation_result["AUC-ROC"]

        pate_scores.append(pate)
        vus_pr_scores.append(vus_pr)
        vus_roc_scores.append(vus_roc)
        auc_pr_scores.append(auc_pr)
        auc_roc_scores.append(auc_roc)

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

    auc_roc_avg = np.mean(auc_roc_scores, dtype=float)
    auc_pr_avg = np.mean(auc_pr_scores, dtype=float)
    vus_roc_avg = np.mean(vus_roc_scores, dtype=float)
    vus_pr_avg = np.mean(vus_pr_scores, dtype=float)
    pate_avg = np.mean(pate_scores, dtype=float)

    auc_roc_std = np.std(auc_roc_scores, dtype=float)
    auc_pr_std = np.std(auc_pr_scores, dtype=float)
    vus_roc_std = np.std(vus_roc_scores, dtype=float)
    vus_pr_std = np.std(vus_pr_scores, dtype=float)
    pate_std = np.std(pate_scores, dtype=float)

    logger.info(f"AUC-ROC: {auc_roc_avg} ± {auc_roc_std}")
    logger.info(f"AUC-PR: {auc_pr_avg} ± {auc_pr_std}")
    logger.info(f"VUS-ROC: {vus_roc_avg} ± {vus_roc_std}")
    logger.info(f"VUS-PR: {vus_pr_avg} ± {vus_pr_std}")
    logger.info(f"PATE: {pate_avg} ± {pate_std}")

    tick_labels = ["PATE", "VUS-PR", "VUS-ROC", "AUC-PR", "AUC-ROC"]
    colors = ["red", "orange", "yellow", "green", "cyan"]
    X = [pate_scores, vus_pr_scores, vus_roc_scores, auc_pr_scores, auc_roc_scores]

    boxplot(X, tick_labels, colors, result_dir)

    return pate_avg
