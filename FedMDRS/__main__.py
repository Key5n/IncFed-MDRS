import os
import numpy as np
from esn import MDRS, RLS
from utils import generate_graph, write_analysis

dataset_folder = "ServerMachineDataset"
files = os.listdir(os.path.join(dataset_folder, "train"))

for file in files:
    print(file)
    data_train = np.genfromtxt(os.path.join(dataset_folder, "train", file), dtype=np.float64, delimiter=",")
    basename = file.split(".")[0]

    N_u = data_train.shape[1]
    N_x = 500
    model = MDRS(N_u, N_x)
    optimizer = RLS(N_x, 1, 1)
    threshold = model.train(data_train, optimizer)

    data_test = np.genfromtxt(os.path.join(dataset_folder, "test", file), dtype=np.float64, delimiter=",")
    label_pred, mahalanobis_distances = model.adapt(data_test, optimizer)

    label_test = np.genfromtxt(os.path.join(dataset_folder, "label_test", file), dtype=np.int64, delimiter=",")

    os.makedirs(f"result/{basename}", exist_ok=True)
    generate_graph(label_test, threshold, mahalanobis_distances, f"result/{basename}/MD.png")
    write_analysis(basename, label_test, label_pred)
