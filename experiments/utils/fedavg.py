from typing import Dict
import numpy as np
import torch


def calc_averaged_weights(state_dicts: list[Dict], data_nums: list[int]):
    averaged_state_dict = {}

    for key in state_dicts[0].keys():
        stacked_tensors = torch.stack(
            [
                data_num * state_dict[key]
                for state_dict, data_num in zip(state_dicts, data_nums)
            ],
        )

        averaged_state_dict[key] = torch.sum(stacked_tensors) / np.sum(data_nums)

    return averaged_state_dict
