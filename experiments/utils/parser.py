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
    args = parser.parse_args()

    return args
