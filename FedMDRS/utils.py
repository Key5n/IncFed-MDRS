import os
import numpy as np
from numpy.typing import NDArray
import warnings
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from esn import MDRS

@dataclass(frozen=True)
class ServerMachineData:
    dataset_name: str
    data_name: str
    data_train: NDArray
    data_test: NDArray
    test_label: NDArray

def create_dataset(
    dataset_name:str = "ServerMachineDataset",
    train_data_dir_path:str = os.path.join("ServerMachineDataset", "train"),
    test_data_dir_path:str = os.path.join("ServerMachineDataset", "test"),
    test_label_dir_path:str = os.path.join("ServerMachineDataset", "test_label"),
) -> list[ServerMachineData]:
    data_filenames = os.listdir(os.path.join(train_data_dir_path))

    dataset: list[ServerMachineData] = []
    for data_filename in data_filenames:
        train_data_file_path = os.path.join(train_data_dir_path, data_filename)
        test_data_file_path = os.path.join(test_data_dir_path, data_filename)
        test_label_file_path = os.path.join(test_label_dir_path, data_filename)

        data_train = np.genfromtxt(train_data_file_path, dtype=np.float64, delimiter=",")
        data_test = np.genfromtxt(test_data_file_path, dtype=np.float64, delimiter=",")
        test_label = np.genfromtxt(test_label_file_path, dtype=np.float64, delimiter=",")

        basename = data_filename.split(".")[0]
        data = ServerMachineData(dataset_name, basename, data_train, data_test, test_label)
        dataset.append(data)

    return dataset

def write_pr_curve(recall, precision, auc, filename):
    plt.clf()

    # Plot Curve
    plt.plot(recall, precision, marker='o', label=f"Precision Recall Curve (AUC = {auc:.4f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    plt.legend()
    plt.savefig(filename)

def write_roc_curve(fprs, tprs, auc, filename):
    plt.clf()

    # Plot Curve
    plt.plot(fprs, tprs, marker='o', label=f"ROC Curve (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(filename)

def update_global_P(P_global: NDArray, local_updates: NDArray):
    top = np.dot(np.dot(P_global, local_updates), P_global)
    bottom = 1 + np.trace(np.dot(local_updates, P_global))
    P_global = P_global - top / bottom

    return P_global

def train_in_clients(serverMachineDataset: list[ServerMachineData], leaking_rate=1.0, rho=0.95, delta=0.0001, input_scale: float = 1.0) -> dict[str,MDRS]:
    models: dict[str, MDRS] = {}
    N_x = 500

    covariance_matrix = np.zeros((N_x, N_x), dtype=np.float64)
    for serverMachineData in serverMachineDataset:
        model, local_updates = train_in_client(serverMachineData, leaking_rate=leaking_rate, rho=rho, delta=delta, input_scale=input_scale)
        models[serverMachineData.data_name] = model

        covariance_matrix += local_updates

    P_global = np.linalg.inv(covariance_matrix + delta * np.identity(N_x))

    for key, model in models.items():
        model.set_P(P_global)
        models[key] = model

    return models

def train_in_client(serverMachineData: ServerMachineData, leaking_rate=1.0, rho=0.95, delta=0.0001, input_scale: float = 1.0) -> tuple[MDRS, NDArray]:
    print(f"[train] data name: {serverMachineData.data_name}")
    data_train = serverMachineData.data_train
    N_u = data_train.shape[1]
    N_x = 500
    model = MDRS(N_u, N_x, leaking_rate=leaking_rate, delta=delta, rho=rho, input_scale=input_scale)
    local_updates = model.train(data_train)

    return model, local_updates

def evaluate_in_clients(models, serverMachineDataset: list[ServerMachineData]) -> tuple[float, float]:
    pr_curve_aucs = []
    pr_curve_aucs_with_modification = []
    for i, serverMachineData in enumerate(serverMachineDataset):
        print(f"Progress Rate: {i / len(serverMachineDataset):.1%}")

        model = models[serverMachineData.data_name]
        pr_curve_auc, pr_curve_auc_with_modification = evaluate_in_client(model, serverMachineData)
        pr_curve_aucs.append(pr_curve_auc)
        pr_curve_aucs_with_modification.append(pr_curve_auc_with_modification)

    return np.mean(pr_curve_aucs, dtype=float), np.mean(pr_curve_aucs_with_modification, dtype=float)

def evaluate_in_client(model, serverMachineData: ServerMachineData, output_dir="result") -> tuple[float, float]:
    name = serverMachineData.data_name
    with open(os.path.join(output_dir, name, "log.txt"), "w") as f:
        print(f"[test] dataset name: {name}", file=f)
        print(f"[test] dataset name: {name}")

        label_test = serverMachineData.test_label
        data_test = serverMachineData.data_test

        _, mahalanobis_distances = model.copy().adapt(data_test)

        precision_modified_label, recall_modified_label, pr_curve_auc_modified_label, fpr_modified_label, tpr_modified_label, roc_curve_auc_modified_label = eval_with_modification(label_test, mahalanobis_distances)
        write_pr_curve(recall_modified_label, precision_modified_label, pr_curve_auc_modified_label, os.path.join(output_dir, name, "pr_curve_with_modification.png"))
        write_roc_curve(fpr_modified_label, tpr_modified_label, roc_curve_auc_modified_label, os.path.join(output_dir, name, "roc_curve_with_modification.png"))

        precision, recall, pr_curve_auc, fpr, tpr, roc_curve_auc = eval_without_modification(label_test, mahalanobis_distances)
        write_pr_curve(recall, precision, pr_curve_auc, os.path.join(output_dir, name, "pr_curve_without_modification.png"))
        write_roc_curve(fpr, tpr, roc_curve_auc, os.path.join(output_dir, name, "roc_curve_without_modification.png"))

        print(f"{roc_curve_auc_modified_label = }, {pr_curve_auc_modified_label = }, {roc_curve_auc = }, {pr_curve_auc = }")
        print(f"{roc_curve_auc_modified_label = }, {pr_curve_auc_modified_label = }, {roc_curve_auc = }, {pr_curve_auc = }", file=f)

        return pr_curve_auc, pr_curve_auc_modified_label

def get_pred_label_for_each_threshold(y_score: NDArray) -> NDArray:
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_sorted = y_score[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_score_sorted.size - 1]
    thresholds = y_score_sorted[threshold_idxs]

    pred = []
    for threshold in thresholds:
        pred_label = y_score >= threshold
        pred.append(pred_label)

    return np.array(pred)

def modify_pred_label(answer_label: NDArray, pred_label: NDArray) -> NDArray:
    # Find the anomaly intervals in the answer_label
    intervals = []
    start = None

    for i, value in enumerate(answer_label):
        if value == 1 and start is None:
            start = i
        elif value == 0 and start is not None:
            intervals.append((start, i))
            start = None

    # Handle the case where the array ends with 1
    if start is not None:
        intervals.append((start, len(answer_label)))

    # Modify predicted_label based on the anomaly intervals
    for start, end in intervals:
        if np.any(pred_label[start:end] == 1):
            pred_label[start:end] = 1

    return np.array(pred_label)

def pr_curve_based_on_label(y_true: NDArray, y_pred_array: NDArray):
    tps = []
    ps = []
    for y_pred in y_pred_array:
        tp = np.sum(y_pred * y_true)
        fp = np.sum((1 - y_true) * y_pred)
        positives = fp + tp

        tps.append(tp)
        ps.append(positives)

    tps = np.array(tps)
    ps = np.array(ps)

    # Initialize the result array with zeros to make sure that precision[ps == 0]
    # does not contain uninitialized values.
    precision = np.zeros_like(tps)
    np.divide(tps, ps, out=precision, where=(ps != 0))

    # When no positive label in y_true, recall is set to 1 for all thresholds
    # tps[-1] == 0 <=> y_true == all negative labels
    if tps[-1] == 0:
        warnings.warn(
            "No positive class found in y_true, "
            "recall is set to one for all thresholds."
        )
        recall = np.ones_like(tps)
    else:
        recall = tps / tps[-1]

    # reverse the outputs so recall is decreasing
    sl = slice(None, None, -1)
    precision = np.hstack((precision[sl], 1))
    recall = np.hstack((recall[sl], 0))
    return precision, recall

def roc_curve_based_on_label(y_true: NDArray, y_pred_array: NDArray):
    tps = []
    fps = []
    ps = []
    for y_pred in y_pred_array:
        # _, fp, _, tp = confusion_matrix(y_true, y_pred).ravel()
        tp = np.sum(y_pred * y_true)
        fp = np.sum((1 - y_true) * y_pred)
        positives = tp + fp

        tps.append(tp)
        fps.append(fp)
        ps.append(positives)

    tps = np.array(tps)
    ps = np.array(ps)

    # Add an extra threshold position
    # to make sure that the curve starts at (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0:
        warnings.warn(
            "No negative samples in y_true, false positive value should be meaningless"
        )
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn(
            "No positive samples in y_true, true positive value should be meaningless"
        )
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    return fpr, tpr

def eval_without_modification(y_true: NDArray, y_score: NDArray) -> tuple[NDArray, NDArray, float, NDArray, NDArray, float]:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    pr_curve_auc = auc(recall, precision)
    roc_curve_auc = auc(fpr, tpr)

    return precision, recall, pr_curve_auc, fpr, tpr, roc_curve_auc

def eval_with_modification(y_true: NDArray, y_score: NDArray) -> tuple[NDArray, NDArray, float, NDArray, NDArray, float]:
    pred_label_for_each_threhold = get_pred_label_for_each_threshold(y_score)
    adjusted_pred_label_for_each_threshold = np.array([modify_pred_label(y_true, pred_label) for pred_label in pred_label_for_each_threhold])

    precision, recall = pr_curve_based_on_label(y_true, adjusted_pred_label_for_each_threshold)
    fpr, tpr = roc_curve_based_on_label(y_true, adjusted_pred_label_for_each_threshold)

    pr_curve_auc = auc(recall, precision)
    roc_curve_auc = auc(fpr, tpr)

    return precision, recall, pr_curve_auc, fpr, tpr, roc_curve_auc
