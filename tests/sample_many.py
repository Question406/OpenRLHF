# from transformers import

import os
import shutil
import click
from pathlib import Path
from datasets import load_dataset, load_from_disk, Dataset
from vllm import LLM, SamplingParams
from typing import Dict, Any
from datetime import datetime
from omegaconf import OmegaConf
from tqdm import tqdm
import copy
import json
import re

from openrlhf.trainer.reward_fns import (
    math_correctness_reward_fn,
    format_reward_fn,
    combined_reward,
)
from openrlhf.utils.logging_utils import init_logger

LOGGER = init_logger(__name__)


def build_dataset(data_path):
    if os.path.exists(data_path):
        dataset = load_from_disk(data_path)
    else:
        dataset = load_dataset(data_path)
    return dataset


def build_vllm(model_name: str):
    llm = LLM(model=model_name)
    return llm


def simplifiy_model_name(model_name):
    return model_name.split("/")[-1]


def samplingParam2json(sampling_param):
    param_str = str(sampling_param)
    # Extract the parameters into a JSON dictionary
    pattern = r"(\w+)=([\[\]{}0-9\.\-]+|True|False|None|'[^']*'|\"[^\"]*\")"
    matches = re.findall(pattern, param_str)
    params_dict = {
        key: eval(value)
        if value not in ["None", "True", "False"]
        else (None if value == "None" else (True if value == "True" else False))
        for key, value in matches
    }
    return params_dict


@click.command()
@click.option("--run_name", prompt="The model name")
@click.option("--model_name", prompt="The model name")
@click.option("--data_path", prompt="The path to data")
@click.option("--prompt_file", prompt="The path to prompt template")
@click.option("--temperature", default=1.0, prompt="The temperature for sampling")
@click.option("--top_p", default=1.0, prompt="The ratio for TopP sampling")
@click.option("--n", default=8, prompt="The number of responses to sample per prompt")
@click.option("--max_tokens", default=2048, prompt="The number of responses to sample per prompt")
@click.pass_context
def main(ctx, **kwarg):
    configs = ctx.params
    assert os.path.exists(configs["prompt_file"]), "The prompt file does not exist"

    dataset = build_dataset(configs["data_path"])
    template_file = open(configs["prompt_file"], "r").read().strip()

    # initialize
    basedir = Path("raw_data") / f"{configs['run_name']}" / "sample_outputs"
    basedir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    now = now.strftime("%m-%d-%H-%M-%S")
    basedir = (
        basedir
        / f"model@{simplifiy_model_name(configs['model_name'])},prompt@{configs['prompt_file'].split('/')[-1].rstrip('.txt')}"
        / now
    )
    basedir.mkdir(parents=True, exist_ok=True)

    sampling_params = SamplingParams(
        temperature=configs["temperature"],
        top_p=configs["top_p"],
        n=configs["n"],
        # seed=42,
        max_tokens=1024,
    )
    # save configs
    out_sampling_params = samplingParam2json(sampling_params)
    configs["sampling_params"] = out_sampling_params
    OmegaConf.save(OmegaConf.create(configs), basedir / "configs.yaml")
    shutil.copy(configs["prompt_file"], basedir / "prompt_template.txt")

    def process_data(sample: Dict[str, Any]):
        # sample["input"] = template_file.format(sample["problem"])
        sample["input"] = template_file.replace("{}", sample["problem"])
        return sample

    # batchify the dataset
    def batchifydata(data, batch_size):
        # yield from [[process_data(x) for x in data[i : i + batch_size]] for i in range(0, len(data), batch_size)]
        yield from [
            [process_data(x) for x in data.select(range(i, min(i + batch_size, len(data))))]
            for i in range(0, len(data), batch_size)
        ]

    format_rewards = []
    math_rewards = []
    combined_rewards = []

    def outputs_generator():
        llm = build_vllm(configs["model_name"])
        for batch in tqdm(batchifydata(dataset, 10)):
            inputs = [x["input"] for x in batch]
            outputs = llm.generate(inputs, sampling_params)
            for sample, output in zip(batch, outputs):
                tmp_sample = copy.deepcopy(sample)
                sample_format_rewards = []
                sample_math_rewards = []
                sample_combined_rewards = []
                for response in output.outputs:
                    response_str = response.text
                    tmp_sample["response"] = response_str
                    tmp_sample["response-id"] = response.index
                    tmp_sample["reward-format"] = format_reward_fn([response_str])
                    tmp_sample["reward-math"] = math_correctness_reward_fn([tmp_sample["gt_answer"]], [response_str])
                    tmp_sample["reward-combine"] = combined_reward([tmp_sample["gt_answer"]], [response_str])

                    sample_format_rewards.append(tmp_sample["reward-format"])
                    sample_math_rewards.append(tmp_sample["reward-math"])
                    sample_combined_rewards.append(tmp_sample["reward-combine"])
                    yield tmp_sample

                format_rewards.append(sample_format_rewards)
                math_rewards.append(sample_math_rewards)
                combined_rewards.append(sample_combined_rewards)

    dataset = Dataset.from_generator(outputs_generator)
    dataset.save_to_disk(str(basedir / "outputs"))

    json.dump(
        {
            "format-reward": format_rewards,
            "math-reward": math_rewards,
            "combined-reward": combined_rewards,
        },
        open(str(basedir / "out-rewards.json"), "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
