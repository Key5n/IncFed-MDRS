from typing import Dict
import numpy as np


def update_client_control_variate(
    current_model: Dict,
    global_model: Dict,
    c_local: Dict,
    c_global: Dict,
    count: int,
    lr: float,
) -> Dict:
    next_c_local = {}
    for key in current_model:
        next_c_local[key] = (
            c_local[key]
            - c_global[key]
            + (global_model[key] - current_model[key]) / (count * lr)
        )

    return next_c_local


def update_model_with_control_variates(
    current_model: Dict, c_global: Dict, c_local: Dict, lr
) -> Dict:
    updated_model = {}

    for key in current_model:
        updated_model[key] = current_model[key] - lr * (c_global[key] - c_local[key])

    return updated_model


def get_client_update(new_c_local: Dict, c_local: Dict) -> Dict:
    c_local_update = {}

    for key in c_local:
        c_local_update[key] = new_c_local[key] - c_local[key]

    return c_local_update


def update_c_global(
    c_global: Dict, client_update_list: list[Dict], num_clients: int
) -> Dict:
    next_c_global = {}
    for key in c_global:
        next_c_global[key] = (
            c_global[key] + np.sum(client_update_list[key]) / num_clients
        )

    return next_c_global
