import os
from utils import train_in_clients, evaluate_in_clients

dirname = "ServerMachineDataset"
train_data_dir_path = os.path.join(dirname, "train")
test_data_dir_path = os.path.join(dirname, "test")
test_label_dir_path = os.path.join(dirname, "test_label")

models_dic = {}

global_precision_matrix = train_in_clients(models_dic, train_data_dir_path)

evaluate_in_clients(global_precision_matrix, models_dic, test_data_dir_path, test_label_dir_path)
