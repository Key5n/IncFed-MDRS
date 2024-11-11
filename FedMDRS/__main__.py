import os
import pickle
import numpy as np
from utils import create_dataset, train_in_clients, train_in_client, evaluate_in_clients, evaluate_in_client

dirname = "ServerMachineDataset"
train_data_dir_path = os.path.join(dirname, "train")
test_data_dir_path = os.path.join(dirname, "test")
test_label_dir_path = os.path.join(dirname, "test_label")

train = True
individually = True
federated = True
save = True

serverMachineDataset = create_dataset()

if federated:
    if train:
        models_dic, P_global = train_in_clients(serverMachineDataset)

        if save:
            print("model and precision matrix are saved")
            with open("models.pickle", "wb") as f:
                pickle.dump(models_dic, f, pickle.HIGHEST_PROTOCOL)
            np.savetxt("P_global.txt", P_global, delimiter=",")
    else:
        P_global = np.genfromtxt("P_global.txt", dtype=np.float64, delimiter=",")
        with open("models.pickle", "rb") as f:
            models_dic = pickle.load(f)

    evaluate_in_clients(P_global, models_dic, serverMachineDataset)

if individually:
    print("individually")
    train_data_filenames = os.listdir(train_data_dir_path)

    for serverMachineData in serverMachineDataset:
        model, _ = train_in_client(serverMachineData)
        evaluate_in_client(model, serverMachineData, output_dir="individual")
