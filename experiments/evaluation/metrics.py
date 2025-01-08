from logging import getLogger
import time
from pate.PATE_metric import PATE
from .basic_metrics import basic_metricor, generate_curve


def get_metrics(score, labels, slidingWindow=100, pred=None, version="opt", thre=250):
    logger = getLogger(__name__)
    metrics = {}

    """
    Threshold Independent
    """
    grader = basic_metricor()
    # AUC_ROC, Precision, Recall, PointF1, PointF1PA, Rrecall, ExistenceReward, OverlapReward, Rprecision, RF, Precision_at_k = grader.metric_new(labels, score, pred, plot_ROC=False)
    auc_roc_start = time.time()
    AUC_ROC = grader.metric_ROC(labels, score)
    auc_roc_end = time.time()
    logger.info(f"AUCROC Time: {auc_roc_end - auc_roc_start}")

    auc_pr_start = time.time()
    AUC_PR = grader.metric_PR(labels, score)
    auc_pr_end = time.time()
    logger.info(f"AUCPR Time: {auc_pr_end - auc_pr_start}")

    # R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
    vus_start = time.time()
    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(
        labels, score, slidingWindow, version, thre
    )
    vus_end = time.time()
    logger.info(f"VUS Time: {vus_end - vus_start}")

    # PATE returns floating[Any] or float, so forces float
    pate_start = time.time()
    pate = float(PATE(labels, score, binary_scores=False, n_jobs=-1))
    pate_end = time.time()
    logger.info(f"PATE Time: {pate_end - pate_start}")

    metrics["AUC-PR"] = AUC_PR
    metrics["AUC-ROC"] = AUC_ROC
    metrics["VUS-PR"] = VUS_PR
    metrics["VUS-ROC"] = VUS_ROC
    metrics["PATE"] = pate

    return metrics
