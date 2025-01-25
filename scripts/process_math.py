# This file contains the scripts to process raw math dataset and save it to disk.

import os
from datasets import Dataset, load_dataset
import json

from openrlhf.trainer.answer_extraction import extract_answer
from collections import Counter


def process_math2disk(raw_data_path: str, out_path: str):
    def build_sample():
        for root, data_split, files in os.walk(raw_data_path):
            for file in files:
                if file.endswith(".json"):
                    with open(os.path.join(root, file), "r") as f:
                        sample = json.load(f)
                    yield {
                        "problem": sample["problem"],
                        "level": sample["level"],
                        "type": sample["type"],
                        "solution": sample["solution"],
                        "gt_answer": extract_answer(sample["solution"]),
                        "uid": file.rstrip(".json"),
                    }

    dataset = Dataset.from_generator(build_sample)
    dataset.save_to_disk(out_path)
    return dataset


def extract_balance_subset(dataset: Dataset, out_path: str, num_samples: int):
    # This funtion extracts a balanced subset of the dataset
    # by keeping the number of problem type equal and the number of problem level equal
    # Count the occurrences of each type and level

    type_counter = Counter(dataset["type"])
    level_counter = Counter(dataset["level"])

    # Determine the minimum count for balanced sampling
    min_type_count = min(type_counter.values())
    min_level_count = min(level_counter.values())

    # Calculate the number of samples per type and level
    samples_per_type = min(num_samples // len(type_counter), min_type_count)
    samples_per_level = min(num_samples // len(level_counter), min_level_count)

    # Collect balanced samples
    balanced_samples = []
    type_samples = {t: [] for t in type_counter}
    level_samples = {level: [] for level in level_counter}

    for sample in dataset:
        if len(type_samples[sample["type"]]) < samples_per_type:
            type_samples[sample["type"]].append(sample)
        if len(level_samples[sample["level"]]) < samples_per_level:
            level_samples[sample["level"]].append(sample)

    for t_samples in type_samples.values():
        balanced_samples.extend(t_samples)
    for l_samples in level_samples.values():
        balanced_samples.extend(l_samples)

    # Shuffle and select the final balanced subset
    # random.shuffle(balanced_samples)
    balanced_subset = balanced_samples[:num_samples]
    # Save the balanced subset to disk
    balanced_dataset = Dataset.from_list(balanced_subset)
    balanced_dataset.save_to_disk(out_path)
    return balanced_dataset


def convert_test(out_path: str):
    def build_sample():
        dataset = load_dataset("json", data_dir="./3rdparty/prm800k/prm800k/math_splits/")["test"]
        # import ipdb

        for idx, sample in enumerate(dataset):
            yield {
                "problem": sample["problem"],
                "level": sample["level"],
                # "type": sample["type"],
                "solution": sample["solution"],
                # "gt_answer": extract_answer(sample["solution"]),
                "gt_answer": sample["answer"],
                "uid": idx,
            }

    # build_sample()

    dataset = Dataset.from_generator(build_sample)
    dataset.save_to_disk(out_path)
    return dataset


# MATH_PATH = "/Users/jiyi/Documents/2025-Projects/OpenRLHF/3rdparty/math/raw_MATH/train"
# OUT_PATH = "raw_data/math_train"

# dataset = process_math2disk(MATH_PATH, OUT_PATH)
# extract_balance_subset(dataset, "raw_data/math_train_balanced-200", 200)

convert_test("./raw_data/math_test")
