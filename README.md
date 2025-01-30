# Incremental Federated Learning with Mahalanobis Distance for Reservoir States (IncFed MD-RS)

# Get Started

## Installation

1. clone this repo
2. run the following commands at root path

```sh
# create python environment
python3 -m venv .venv/bin/activate
. .venv/bin/activate
pip3 install -r requirements.txt

# get all datasets
# For SMAP, make sure you have an Kaggle API key setup, then:
sh get_dataset.sh
```

We use the version 3.10 of python3.

## Usage

To run IncFed MD-RS, run the following command.

```sh
python3 fedmdrs_main.py --dataset [(SMD)|(SMAP)|(PSM)]
```

For example, run the following command for SMD (Server Machine Dataset).

```sh
python3 fedmdrs_main.py --dataset SMD
```
