# This file contains tool functions for processing gsm8k dataset.

import numpy as np
from datasets import load_dataset, load_from_disk, Dataset
from collections import defaultdict
from openrlhf.trainer.answer_extraction import extract_answer
from openrlhf.trainer.eval_utils import math_equal


def convert_gsm8k(save_path):
    dataset = load_dataset("openai/gsm8k", "main")["train"]

    dataset = dataset.rename_column("question", "problem")
    dataset = dataset.rename_column("answer", "solution")

    def add_gt_answer(sample, index):
        sample["gt_answer"] = extract_answer(sample["solution"])
        sample["uid"] = index
        return sample

    dataset = dataset.map(add_gt_answer, with_indices=True)
    dataset.save_to_disk(save_path)
    return dataset


def filter_hard_question(load_path, save_path, threshold):
    dataset = load_from_disk(load_path)
    origin_dataset = load_dataset("openai/gsm8k", "main")["train"]
    # count accuracy for each uid
    accuracy_mapping = defaultdict(list)
    for sample in dataset:
        if math_equal(sample["response"], sample["gt_answer"]):
            accuracy_mapping[sample["uid"]].append(1)
        else:
            accuracy_mapping[sample["uid"]].append(0)

    accuracy_mapping = {k: np.mean(v) for k, v in accuracy_mapping.items()}
    filterd_question_ids = {k for k, v in accuracy_mapping.items() if v > threshold}
    # filter original dataset with the accuracy
    filterd_dataset = origin_dataset.filter(lambda x: x["uid"] in filterd_question_ids)
    filterd_dataset.save_to_disk(save_path)
    return filterd_dataset


def get_accuracy_mapping(dataset: Dataset):
    accuracy_mapping = defaultdict(list)
    for sample in dataset:
        if math_equal(sample["response"], sample["gt_answer"]):
            accuracy_mapping[sample["uid"]].append(1)
        else:
            accuracy_mapping[sample["uid"]].append(0)
    return accuracy_mapping


def pass_at_k(accmapping, k):
    res = {}
    for key, value in accmapping.items():
        res[key] = np.mean(np.mean([value[i * k : (i + 1) * k] for i in range(0, len(value), k)]))
    return np.mean(list(res.values()))


if __name__ == "__main__":
    # convert_gsm8k("gsm8k")
    filter_hard_question("gsm8k", "gsm8k_hard", 0.8)
