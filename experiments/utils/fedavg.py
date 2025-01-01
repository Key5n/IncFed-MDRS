from typing import Dict
import numpy as np
import torch


def calc_averaged_weights(state_dicts: list[Dict], data_nums: list[int]):
    averaged_state_dict = {}

    data_nums_ratio = data_nums / np.sum(data_nums)

    for key in state_dicts[0].keys():
        stacked_params = torch.stack(
            [
                data_num_ratio * state_dict[key]
                for state_dict, data_num_ratio in zip(state_dicts, data_nums_ratio)
            ]
        )

        averaged_state_dict[key] = torch.sum(stacked_params, dim=0)

    return averaged_state_dict
