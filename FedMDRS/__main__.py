import os
from esn import  RLS
from utils import train_in_clients, evaluate_in_clients

dirname = "ServerMachineDataset"
train_data_dir_path = os.path.join(dirname, "train")
test_data_dir_path = os.path.join(dirname, "test")
test_label_dir_path = os.path.join(dirname, "test_label")

N_x = 500
global_optimizer = RLS(N_x, 0.0001, 1)
models_dic = {}

train_in_clients(global_optimizer, models_dic, train_data_dir_path)

evaluate_in_clients(global_optimizer, models_dic, test_data_dir_path, test_label_dir_path)
