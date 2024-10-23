import os
import numpy as np
from esn import MDRS, RLS
from utils import train_in_clients, evaluate_in_clients
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, auc

dirname = "ServerMachineDataset"
train_data_dir_path = os.path.join(dirname, "train")
test_data_dir_path = os.path.join(dirname, "test")
test_label_dir_path = os.path.join(dirname, "test_label")

N_x = 500
global_optimizer = RLS(N_x, 0.0001, 1)
models_dic = {}

train_in_clients(global_optimizer, models_dic, train_data_dir_path)

evaluate_in_clients(global_optimizer, models_dic, test_data_dir_path, test_label_dir_path)

    # data_train = np.genfromtxt(os.path.join(dirname,  filename), dtype=np.float64, delimiter=",")
    # basename = filename.split(".")[0]
    #
    # N_u = data_train.shape[1]
    # model = MDRS(N_u, N_x)
    # optimizer = RLS(N_x, 0.00001, 1)
    # threshold = model.train(data_train, optimizer, global_optimizer)
    #
    # data_test = np.genfromtxt(os.path.join(, "test", filename), dtype=np.float64, delimiter=",")
    #
    # label_preds = []
    # mahalanobis_distances_list = []
    #
    # false_positive_rates = [1]
    # true_positive_rates = [1]
    # threshold = 0
    # label_test = np.genfromtxt(os.path.join(dataset_folder, "test_label", filename), dtype=np.int64, delimiter=",")
    # while false_positive_rates[-1] != 0 and true_positive_rates[-1] != 0:
    #     print(f"*** {threshold = } ***")
    #     model_copied = model.copy()
    #     optimizer_copied = global_optimizer.copy()
    #     label_pred, mahalanobis_distances = model_copied.adapt(data_test, optimizer_copied.copy(), threshold)
    #     cm = confusion_matrix(label_test, label_pred)
    #     print(f"{cm = }")
    #     tn, fp, fn, tp = cm.flatten()
    #     fpr = fp / (fp + tn)
    #     tpr = tp / (tp + fn)
    #     print(f"{tpr = }, {fpr = }")
    #     false_positive_rates.append(fpr)
    #     true_positive_rates.append(tpr)
    #
    #     if threshold <= 0.15:
    #         threshold += 0.025
    #     elif threshold <= 1.0:
    #         threshold += 0.1
    #     else:
    #         threshold *= 2
    #
    # os.makedirs(f"result/{basename}", exist_ok=True)
    # # generate_graph(label_test, threshold, mahalanobis_distances, f"result/{basename}/MD.png")
    # # write_analysis(basename, label_test, label_pred)
    # roc_auc = auc(false_positive_rates, true_positive_rates)
    # print(f"{roc_auc = }")
    # write_roc_curve(false_positive_rates, true_positive_rates, roc_auc, f"result/{basename}/roc.png")
    #
    #
    #
