from pate.PATE_metric import PATE
from .basic_metrics import basic_metricor, generate_curve


def get_metrics(score, labels, slidingWindow=100, pred=None, version="opt", thre=250):
    metrics = {}

    """
    Threshold Independent
    """
    grader = basic_metricor()
    # AUC_ROC, Precision, Recall, PointF1, PointF1PA, Rrecall, ExistenceReward, OverlapReward, Rprecision, RF, Precision_at_k = grader.metric_new(labels, score, pred, plot_ROC=False)
    AUC_ROC = grader.metric_ROC(labels, score)
    AUC_PR = grader.metric_PR(labels, score)

    # R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(
        labels, score, slidingWindow, version, thre
    )

    # PATE returns floating[Any] or float, so forces float
    pate = float(PATE(labels, score, binary_scores=False, n_jobs=-1))

    metrics["AUC-PR"] = AUC_PR
    metrics["AUC-ROC"] = AUC_ROC
    metrics["VUS-PR"] = VUS_PR
    metrics["VUS-ROC"] = VUS_ROC
    metrics["PATE"] = pate

    return metrics
