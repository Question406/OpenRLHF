# This file contains function to convert countdown dataset

from datasets import load_dataset, Dataset


def convert_countdown(out_path):
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

    def build_sample():
        for sample in dataset:
            yield {
                "problem": str(sample["nums"]),
                "solution": sample["target"],
                "target": sample["target"],
                "gt_answer": "{nums} = {target}".format(**sample),
            }

    newdata = Dataset.from_generator(build_sample)
    newdata.save_to_disk(out_path)


convert_countdown("raw_data/countdown_train")
