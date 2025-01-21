import os
from tqdm.contrib import tenumerate
from torch.utils.data import DataLoader

from experiments.algorithms.USAD.utils import getting_labels
from experiments.evaluation.metrics import get_metrics
from experiments.utils.diagram.plot import plot
from experiments.utils.get_final_scores import get_final_scores


def evaluate(model, test_dataloader_list: list[DataLoader], result_dir: str):
    evaluation_results = []
    for i, test_dataloader in tenumerate(test_dataloader_list):
        scores = model.copy().run(test_dataloader)
        labels = getting_labels(test_dataloader)

        plot(scores, labels, os.path.join(result_dir, f"{i}.pdf"))

        evaluation_result = get_metrics(scores, labels)
        evaluation_results.append(evaluation_result)

    pate_avg = get_final_scores(evaluation_results, result_dir)

    return pate_avg
