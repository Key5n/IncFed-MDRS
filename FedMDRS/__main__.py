import os
import numpy as np
from esn import MDRS, RLS
from utils import generate_graph, write_analysis, write_roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, auc

dataset_folder = "ServerMachineDataset"
files = os.listdir(os.path.join(dataset_folder, "train"))

for file in files:
    print(file)

    data_train = np.genfromtxt(os.path.join(dataset_folder, "train", file), dtype=np.float64, delimiter=",")
    basename = file.split(".")[0]

    N_u = data_train.shape[1]
    N_x = 500
    model = MDRS(N_u, N_x)
    optimizer = RLS(N_x, 0.00001, 1)
    threshold = model.train(data_train, optimizer)

    data_test = np.genfromtxt(os.path.join(dataset_folder, "test", file), dtype=np.float64, delimiter=",")

    label_preds = []
    mahalanobis_distances_list = []

    false_positive_rates = [1]
    true_positive_rates = [1]
    threshold = 0
    label_test = np.genfromtxt(os.path.join(dataset_folder, "test_label", file), dtype=np.int64, delimiter=",")
    while false_positive_rates[-1] != 0 and true_positive_rates[-1] != 0:
        print(f"*** {threshold = } ***")
        model_copied = model.copy()
        optimizer_copied = optimizer.copy()
        label_pred, mahalanobis_distances = model_copied.adapt(data_test, optimizer_copied.copy(), threshold)
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

