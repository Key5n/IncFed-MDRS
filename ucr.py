import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from FedMDRS import esn, utils

file = "141_UCR_Anomaly_InternalBleeding5_4000_6200_6370.txt"

# data_train = np.genfromtxt(os.path.join(dataset_folder, "train", file), dtype=np.float64, delimiter=",")
data_train = np.genfromtxt(file, dtype=np.float64, delimiter=",")[:4000].reshape((-1, 1))
basename = file.split(".")[0]

mmscaler = MinMaxScaler()
mmscaler.fit(data_train)
data_train_scaled = mmscaler.transform(data_train)
data_train_scaled = np.reshape(data_train_scaled, (-1,1))

N_u = data_train_scaled.shape[1]
N_x = 500
model = esn.MDRS(N_u, N_x)
optimizer = esn.RLS(N_x, 0.00001, 1)
threshold = model.train(data_train_scaled, optimizer)

print(f"{threshold =}")

data_test = np.genfromtxt(file, dtype=np.float64, delimiter=",")[4001:].reshape((-1,1))
mmscaler = MinMaxScaler()
mmscaler.fit(data_test)
data_test_scaled = mmscaler.transform(data_test)
data_test_scaled = np.reshape(data_test_scaled, (-1,1))


print(f"{data_test_scaled=}")
label_pred, mahalanobis_distances = model.adapt(data_test_scaled, optimizer)

print(mahalanobis_distances.shape, label_pred.shape)

# label_test = np.genfromtxt(os.path.join(dataset_folder, "label_test", file), dtype=np.int64, delimiter=",")
label_test = np.zeros(len(data_test))
label_test[6200-4001:6370-4001] = 1

os.makedirs(f"result/{basename}", exist_ok=True)
utils.generate_graph(label_test, threshold, mahalanobis_distances, f"result/{basename}/MD.png")
utils.write_analysis(basename, label_test, label_pred)

