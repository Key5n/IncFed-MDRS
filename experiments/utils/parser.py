import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        metavar="-d",
        type=str,
        required=True,
        choices=["SMD", "SMAP", "PSM"],
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=[
            "FedMDRS",
            "FedAvg_MDRS",
            "IncFed",
            "TranAD",
            "LSTMAE",
            "FedAvg_TranAD",
            "FedAvg_LSTMAE",
        ],
    )
    args = parser.parse_args()

    return args
