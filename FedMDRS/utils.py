import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report, precision_score, recall_score, auc, roc_curve, precision_recall_curve
from esn import MDRS

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

def train_in_clients(models, train_data_dir_path):
    train_data_filenames = os.listdir(train_data_dir_path)
    N_x = 500
    delta = 0.0001
    P_global = (1.0 / delta) * np.eye(N_x, N_x)
    P_global_next = P_global

    for train_data_filename in train_data_filenames:
        train_data_file_path = os.path.join(train_data_dir_path, train_data_filename)
        P_global_next = train_in_client(P_global_next, models, train_data_file_path)

    return P_global_next

def train_in_client(P_global, models, train_data_file_path):
    print(f"train {train_data_file_path = }")
    data_train = np.genfromtxt(train_data_file_path, dtype=np.float64, delimiter=",")

    N_u = data_train.shape[1]
    N_x = 500
    model = MDRS(N_u, N_x)
    P_global_next = model.train(data_train, P_global)

    filename = train_data_file_path.split("/")[-1]
    basename = filename.split(".")[0]

    models[basename] = model

    return P_global_next

def evaluate_in_clients(P_global, models, test_data_dir_path, test_label_dir_path):
    test_data_filenames = os.listdir(test_data_dir_path)

    for i, test_data_filename in enumerate(test_data_filenames):
        basename = test_data_filename.split(".")[0]
        model = models[basename]
        test_data_file_path = os.path.join(test_data_dir_path, test_data_filename)
        test_label_file_path = os.path.join(test_label_dir_path, test_data_filename)

        print(f"Progress Rate: {i / len(test_data_filenames) * 100}%")

        evaluate_in_client(P_global, model, test_data_file_path, test_label_file_path)

def evaluate_in_client(P_global, model, test_data_file_path, test_label_file_path):
    basename = test_data_file_path.split(".")[0].split("/")[-1]
    data_test = np.genfromtxt(test_data_file_path, dtype=np.float64, delimiter=",")
    with open(f"result/{basename}/log.txt", "w") as f:
        print(f"test {test_label_file_path = }", file=f)
        print(f"test {test_label_file_path = }")

        false_positive_rates = [1]
        true_positive_rates = [1]
        precision_scores = [0]

        threshold = 0
        label_test = np.genfromtxt(test_label_file_path, dtype=np.int64, delimiter=",")
        while false_positive_rates[-1] != 0 or true_positive_rates[-1] != 0:
            print(f"*** {threshold = } ***", file=f)
            print(f"*** {threshold = } ***")
            label_pred, mahalanobis_distances = model.copy().adapt(data_test, precision_matrix=P_global.copy(), threshold=threshold)
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

            false_positive_rates.append(fpr)
            true_positive_rates.append(tpr)
            precision_scores.append(precision)

            if threshold <= 0.1:
                threshold += 0.001
            elif threshold <= 1.0:
                threshold += 0.1
            else:
                threshold *= 2

        os.makedirs(f"result/{basename}", exist_ok=True)
        # generate_graph(label_test, threshold, mahalanobis_distances, f"result/{basename}/MD.png")
        # write_analysis(basename, label_test, label_pred)
        roc_auc = auc(false_positive_rates, true_positive_rates)
        precision_recall_curve_auc = auc(true_positive_rates, precision_scores)

        print(f"{roc_auc = }, {precision_recall_curve_auc = }")
        print(f"{roc_auc = }, {precision_recall_curve_auc = }", file=f)

        write_curve(false_positive_rates, true_positive_rates, roc_auc, f"result/{basename}/roc.png", name="ROC")
        write_curve(precision_scores, true_positive_rates, precision_recall_curve_auc, f"result/{basename}/precision_recall.png")
