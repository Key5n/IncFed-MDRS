import os
import numpy as np
from numpy.typing import NDArray
from collections import Counter
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix, f1_score, classification_report, precision_score, recall_score, auc, precision_recall_curve, roc_curve
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
    train_data_dir_path:str = "ServerMachineDataset/train",
    test_data_dir_path:str = "ServerMachineDataset/test",
    test_label_dir_path:str = "ServerMachineDataset/test_label",
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

def write_curve(y_array, x_array, auc, filename, name="Precision Recall"):
    # name should be "ROC" or "Precision Recall"
    plt.clf()

    # Plot Curve
    plt.plot(y_array, x_array, marker='o', label=f"{name} Curve (AUC = {auc:.4f})")

    if name == "ROC":
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Model")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} Curve")
    plt.legend()
    plt.savefig(filename)

def train_in_clients(serverMachineDataset: list[ServerMachineData]) -> tuple[dict[str,MDRS], NDArray]:
    models: dict[str, MDRS] = {}
    N_x = 500
    delta = 0.0001
    P_global = (1.0 / delta) * np.eye(N_x, N_x)
    P_global_next = P_global

    for serverMachineData in serverMachineDataset:
        model, P_global_next = train_in_client(serverMachineData, P=P_global_next)
        models[serverMachineData.data_name] = model

    if P_global_next is None:
        raise ValueError("The type of Precision Matrix is None")

    return models, P_global_next

def train_in_client(serverMachineData: ServerMachineData, P: NDArray | None = None) -> tuple[MDRS, NDArray | None]:
    print(f"[train] data name: {serverMachineData.data_name}")
    data_train = serverMachineData.data_train
    N_u = data_train.shape[1]
    N_x = 500
    model = MDRS(N_u, N_x)

    if P is None:
        model.train(data_train)
        return model, None
    else:
        P_global_next = model.train(data_train, P_global=P)
        return model, P_global_next

def evaluate_in_clients(P_global, models, serverMachineDataset: list[ServerMachineData]) -> None:
    for i, serverMachineData in enumerate(serverMachineDataset):
        print(f"Progress Rate: {i / len(serverMachineDataset) * 100}%")

        model = models[serverMachineData.data_name]
        evaluate_in_client(model, serverMachineData, P=P_global)

def evaluate_in_client(model, serverMachineData: ServerMachineData, P:NDArray | None=None, output_dir="result") -> None:
    name = serverMachineData.data_name
    os.makedirs(f"{output_dir}/{name}", exist_ok=True)
    with open(f"{output_dir}/{name}/log.txt", "w") as f:
        print(f"[test] dataset name: {name}", file=f)
        print(f"[test] dataset name: {name}")

        label_test = serverMachineData.test_label
        data_test = serverMachineData.data_test

        if P is not None:
            _, mahalanobis_distances = model.copy().adapt(data_test, precision_matrix=P.copy())
        else:
            _, mahalanobis_distances = model.copy().adapt(data_test)

        precision, recall, _ = precision_recall_curve(label_test, mahalanobis_distances)
        fpr, tpr, _ = roc_curve(label_test, mahalanobis_distances)

        precision_recall_curve_auc = auc(recall, precision)
        roc_curve_auc = auc(fpr, tpr)
        write_curve(recall, precision, precision_recall_curve_auc,  f"{output_dir}/{name}/precision_recall.png")
        write_curve(fpr, tpr, roc_curve_auc,  f"{output_dir}/{name}/roc_curve.png")

        print(f"{roc_curve_auc = }, {precision_recall_curve_auc = }")
        print(f"{roc_curve_auc = }, {precision_recall_curve_auc = }", file=f)

        value_counts = Counter(mahalanobis_distances)
        for value, count in value_counts.items():
            print(f"Value {value}: {count}")

def create_anomaly_sequences(label_test: NDArray) -> list[list[int]]:
    intervals: list[list[int]] = []
    start: int | None = None

    for i, value in enumerate(label_test):
        if value == 1 and start is None:
            # Start of a new interval
            start = i
        elif value == 0 and start is not None:
            # End of the current interval
            intervals.append([start, i])
            start = None

    # Handle the case where the array ends with 1
    if start is not None:
        intervals.append([start, len(label_test)])

    return intervals

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
