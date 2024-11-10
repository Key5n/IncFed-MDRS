import os
import pickle
import numpy as np
from utils import train_in_clients, evaluate_in_clients

dirname = "ServerMachineDataset"
train_data_dir_path = os.path.join(dirname, "train")
test_data_dir_path = os.path.join(dirname, "test")
test_label_dir_path = os.path.join(dirname, "test_label")

train = True
save = True

if train:
    models_dic = {}
    P_global = train_in_clients(models_dic, train_data_dir_path)

    if save:
        with open("models.pickle", "wb") as f:
            pickle.dump(models_dic, f, pickle.HIGHEST_PROTOCOL)
        np.savetxt("P_global.txt", P_global, delimiter=",")
else:
    P_global = np.genfromtxt("P_global.txt", dtype=np.float64, delimiter=",")
    with open("models.pickle", "rb") as f:
        models_dic = pickle.load(f)

evaluate_in_clients(P_global, models_dic, test_data_dir_path, test_label_dir_path)
