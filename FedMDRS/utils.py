import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report, precision_score, recall_score, auc, roc_curve
from esn import MDRS, RLS

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

def write_roc_curve(false_positives, true_positives, roc_auc, filename):
    plt.clf()
   # Plot ROC Curve
    plt.plot(false_positives, true_positives, marker='o', label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(filename)

# Specify the files you want
target_filenames = ["machine-1-5.txt", "machine-1-1.txt", "machine-1-2.txt"]

def train_in_clients(global_optimizer, models, train_data_dir_path):
    train_data_filenames = os.listdir(train_data_dir_path)[:15]
   # Specify the files you want
    # target_filenames = ["machine-1-5.txt", "machine-1-1.txt", "machine-1-2.txt"]

    # Filter the original list
    # filtered_filenames = [filename for filename in train_data_filenames if filename in target_filenames]

    for train_data_filename in train_data_filenames:
    # for train_data_filename in filtered_filenames:
        train_data_file_path = os.path.join(train_data_dir_path, train_data_filename)
        train_in_client(global_optimizer, models, train_data_file_path)

def train_in_client(global_optimizer, models, train_data_file_path):
    print(f"train {train_data_file_path = }")
    data_train = np.genfromtxt(train_data_file_path, dtype=np.float64, delimiter=",")

    N_u = data_train.shape[1]
    N_x = 500
    model = MDRS(N_u, N_x)
    optimizer = RLS(N_x, 0.00001, 1)
    model.train(data_train, optimizer, global_optimizer)

    filename = train_data_file_path.split("/")[-1]
    basename = filename.split(".")[0]

    models[basename] = model
    print(models)

def evaluate_in_clients(global_optimizer, models, test_data_dir_path, test_label_dir_path):
    test_data_filenames = os.listdir(test_data_dir_path)[:15]

    # Filter the original list
    # filtered_filenames = [filename for filename in test_data_filenames if filename in target_filenames]

    for test_data_filename in test_data_filenames:
    # for test_data_filename in filtered_filenames:
        basename = test_data_filename.split(".")[0]
        model = models[basename]
        test_data_file_path = os.path.join(test_data_dir_path, test_data_filename)
        test_label_file_path = os.path.join(test_label_dir_path, test_data_filename)

        evaluate(global_optimizer, model, test_data_file_path, test_label_file_path)

def evaluate(global_optimizer, model, test_data_file_path, test_label_file_path):
    print(f"test {test_label_file_path = }")
    data_test = np.genfromtxt(test_data_file_path, dtype=np.float64, delimiter=",")
    basename = test_data_file_path.split(".")[0].split("/")[-1]

    false_positive_rates = [1]
    true_positive_rates = [1]
    threshold = 0
    label_test = np.genfromtxt(test_label_file_path, dtype=np.int64, delimiter=",")
    while false_positive_rates[-1] != 0 or true_positive_rates[-1] != 0:
        print(f"*** {threshold = } ***")
        label_pred, mahalanobis_distances = model.copy().adapt(data_test, global_optimizer.copy(), threshold)
        cm = confusion_matrix(label_test, label_pred)
        print(f"{cm = }")
        tn, fp, fn, tp = cm.flatten()
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        print(f"{tpr = }, {fpr = }")
        false_positive_rates.append(fpr)
        true_positive_rates.append(tpr)

        if threshold <= 0.15:
            threshold += 0.025
        elif threshold <= 1.0:
            threshold += 0.1
        else:
            threshold *= 2

    os.makedirs(f"result/{basename}", exist_ok=True)
    # generate_graph(label_test, threshold, mahalanobis_distances, f"result/{basename}/MD.png")
    # write_analysis(basename, label_test, label_pred)
    roc_auc = auc(false_positive_rates, true_positive_rates)
    print(f"{roc_auc = }")
    write_roc_curve(false_positive_rates, true_positive_rates, roc_auc, f"result/{basename}/roc.png")
