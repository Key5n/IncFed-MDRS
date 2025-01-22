from logging import getLogger
from tqdm import tqdm
import numpy as np
from typing import Dict
from torch.utils.data import DataLoader

from experiments.algorithms.USAD.utils import getting_labels
from experiments.evaluation.metrics import get_metrics


def evaluate(model, test_dataloader_list: list[DataLoader]) -> Dict:
    logger = getLogger(__name__)
    anomaly_scores = []
    labels_list = []
    auc_roc_scores = []
    auc_pr_scores = []
    vus_roc_scores = []
    vus_pr_scores = []
    pate_scores = []

    for test_dataloader in tqdm(test_dataloader_list):
        scores = model.copy().run(test_dataloader)
        labels = getting_labels(test_dataloader)

        evaluation_result = get_metrics(scores, labels)

        auc_roc = evaluation_result["AUC-ROC"]
        auc_pr = evaluation_result["AUC-PR"]
        vus_roc = evaluation_result["VUS-ROC"]
        vus_pr = evaluation_result["VUS-PR"]
        pate = evaluation_result["PATE"]

        anomaly_scores.append(scores)
        labels_list.append(labels)

        auc_roc_scores.append(auc_roc)
        auc_pr_scores.append(auc_pr)
        vus_roc_scores.append(vus_roc)
        vus_pr_scores.append(vus_pr)
        pate_scores.append(pate)

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

    result = {
        "anomaly_scores": anomaly_scores,
        "labels_list": labels_list,
        "auc_roc_scores": auc_roc_scores,
        "auc_pr_scores": auc_pr_scores,
        "vus_roc_scores": vus_roc_scores,
        "vus_pr_scores": vus_pr_scores,
        "pate_scores": pate_scores,
    }

    return result
