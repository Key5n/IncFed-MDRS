import os
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix, f1_score, classification_report, precision_score, recall_score, auc, roc_curve, precision_recall_curve
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

        false_positive_rates = []
        true_positive_rates = []
        precision_scores = []

        threshold = 0
        label_test = serverMachineData.test_label
        data_test = serverMachineData.data_test
        while len(true_positive_rates) == 0 or true_positive_rates[-1] != 0:
            print(f"*** {threshold = } ***", file=f)
            print(f"*** {threshold = } ***")

            if P is not None:
                label_pred, _ = model.copy().adapt(data_test, precision_matrix=P.copy(), threshold=threshold)
            else:
                label_pred, _ = model.copy().adapt(data_test,  threshold=threshold)

            cm = confusion_matrix(label_test, label_pred)
            tn, fp, fn, tp = cm.flatten()

            # recall is the same as true positive rate
            fpr = fp / (fp + tn)
            tpr = tp / (tp + fn)
            precision = tp / (tp + fp)

            print(f"{cm = }", file=f)
            print(f"{cm = }")
            print(f"{tpr = }, {fpr = }, {precision = }", file=f)
            print(f"{tpr = }, {fpr = }, {precision = }")

            last_tpr = true_positive_rates[-1] if len(true_positive_rates) != 0 else 1
            diff = np.abs(tpr - last_tpr)

            increment = 0.0001

            if diff >= 0.1:
                increment /= 2
                threshold = threshold - increment
                print(f"{bcolors.FAIL}Over{bcolors.ENDC}")
                print(f"{bcolors.FAIL}Over{bcolors.ENDC}", file=f)
            elif diff <= 0.005:
                increment *= 2
                threshold += increment
                print(f"{bcolors.WARNING}Too Small{bcolors.ENDC}")
                print(f"{bcolors.WARNING}Too Small{bcolors.ENDC}", file=f)
            else:
                false_positive_rates.append(fpr)
                true_positive_rates.append(tpr)
                precision_scores.append(precision)

                threshold += increment
                print(f"{bcolors.OKGREEN}Added{bcolors.ENDC}")
                print(f"{bcolors.OKGREEN}Added{bcolors.ENDC}", file=f)

        # generate_graph(label_test, threshold, mahalanobis_distances, f"{output_dir}/{basename}/MD.png")
        # write_analysis(basename, label_test, label_pred)
        roc_auc = auc(false_positive_rates, true_positive_rates)
        precision_recall_curve_auc = auc(true_positive_rates, precision_scores)

        print(f"{roc_auc = }, {precision_recall_curve_auc = }")
        print(f"{roc_auc = }, {precision_recall_curve_auc = }", file=f)

        write_curve(false_positive_rates, true_positive_rates, roc_auc, f"{output_dir}/{name}/roc.png", name="ROC")
        write_curve(precision_scores, true_positive_rates, precision_recall_curve_auc, f"{output_dir}/{name}/precision_recall.png")

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
