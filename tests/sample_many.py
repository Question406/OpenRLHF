# from transformers import

import os
import shutil
import click
from pathlib import Path
from datasets import load_dataset, load_from_disk, Dataset
from vllm import LLM, SamplingParams
from datetime import datetime
from omegaconf import OmegaConf
from tqdm import tqdm
import copy
import json
import re
from typing import Dict, Any
from transformers import AutoTokenizer

from openrlhf.trainer.reward_fns import (
    math_correctness_reward_fn,
    format_reward_fn,
    combined_reward,
)
from openrlhf.utils.logging_utils import init_logger
from openrlhf.datasets.prompts_dataset import preprocess_data, load_constants

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
# @click.option("--stop", default=None, prompt="List of stop words", multiple=True)
# @click.pass_context
def main(**kwargs):
    # ctx = kwargs
    # configs = ctx.params
    configs = kwargs
    configs["stop"] = None
    if configs["stop"] is not None:
        configs["stop"] = list(configs["stop"])

    assert os.path.exists(configs["prompt_file"]), "The prompt file does not exist"

    dataset = build_dataset(configs["data_path"])
    if configs["prompt_file"].endswith(".py"):
        load_constants(configs["prompt_file"])
        template_file = None
    elif configs["prompt_file"].endswith(".txt"):
        template_file = open(configs["prompt_file"], "r").read().strip()
    else:
        raise ValueError("The prompt file should be either a .py or .txt file")

    # initialize
    basedir = Path("raw_data") / f"{configs['run_name']}" / "sample_outputs"
    basedir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    now = now.strftime("%m-%d-%H-%M-%S")
    basedir = (
        basedir
        / f"model@{simplifiy_model_name(configs['model_name'])},prompt@{configs['prompt_file'].split('/')[-1].rstrip('.txt').rstrip('.py')}"
        / now
    )
    basedir.mkdir(parents=True, exist_ok=True)

    # seed=42,
    sampling_params = SamplingParams(
        temperature=configs["temperature"],
        top_p=configs["top_p"],
        n=configs["n"],
        max_tokens=2048,
        stop=configs["stop"],
    )
    # save configs
    out_sampling_params = samplingParam2json(sampling_params)
    configs["sampling_params"] = out_sampling_params
    OmegaConf.save(OmegaConf.create(configs), basedir / "configs.yaml")
    shutil.copy(configs["prompt_file"], basedir / "prompt_template.txt")

    tokenizer = AutoTokenizer.from_pretrained(configs["model_name"])

    # def process_data(sample: Dict[str, Any]):
    #     # sample["input"] = template_file.format(sample["problem"])
    #     sample["input"] = template_file.replace("{}", sample["problem"])
    #     return sample
    def process_sample(sample: Dict[str, Any], input):
        sample["input"] = input
        return sample

    # batchify the dataset
    def batchifydata(data, batch_size):
        # yield from [[process_data(x) for x in data[i : i + batch_size]] for i in range(0, len(data), batch_size)]
        yield from [
            # [process_data(x) for x in data.select(range(i, min(i + batch_size, len(data))))]
            [
                process_sample(
                    x,
                    preprocess_data(
                        x,
                        input_template=template_file,
                        input_key="problem",
                        apply_chat_template=tokenizer.apply_chat_template,
                    ),
                )
                for x in data.select(range(i, min(i + batch_size, len(data))))
            ]
            for i in range(0, len(data), batch_size)
        ]

    format_rewards = []
    math_rewards = []
    combined_rewards = []
    all_res = []

    def outputs_generator():
        llm = build_vllm(configs["model_name"])
        for batch in tqdm(batchifydata(dataset, 10)):
            inputs = [x["input"] for x in batch]
            # inputs = batch
            outputs = llm.generate(inputs, sampling_params)
            for sample, output in zip(batch, outputs):
                sample_format_rewards = []
                sample_math_rewards = []
                sample_combined_rewards = []
                for response in output.outputs:
                    tmp_sample = copy.deepcopy(sample)
                    response_str = response.text
                    tmp_sample["response"] = response_str
                    tmp_sample["response-id"] = response.index
                    tmp_sample["reward-format"] = format_reward_fn([response_str])[0]
                    tmp_sample["reward-math"] = math_correctness_reward_fn([tmp_sample["gt_answer"]], [response_str])[
                        0
                    ]
                    tmp_sample["reward-combine"] = combined_reward([tmp_sample["gt_answer"]], [response_str])[0]

                    sample_format_rewards.append(tmp_sample["reward-format"])
                    sample_math_rewards.append(tmp_sample["reward-math"])
                    sample_combined_rewards.append(tmp_sample["reward-combine"])
                    all_res.append(tmp_sample)

                format_rewards.append(sample_format_rewards)
                math_rewards.append(sample_math_rewards)
                combined_rewards.append(sample_combined_rewards)

    # dataset = Dataset.from_generator(outputs_generator)
    # all_res = outputs_generator()
    outputs_generator()
    dataset = Dataset.from_list(all_res)
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
