import os
import sys
import copy
import importlib
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List, Dict

# This is a declaration of the MESSAGE_TEMPLATE variable
# In training, load_constants function may load a python file that contains this variable, and overwrites this None value
MESSAGE_TEMPLATE = None


class CustomDefaultDict(dict):
    def __missing__(self, key):
        return f"{{{key}}}"  # Keep the placeholder as-is


def apply_prompt(prompt_list: List[Dict[str, str]], replace_dict: Dict[str, str]) -> List[Dict[str, str]]:
    prompt_list = copy.deepcopy(prompt_list)
    prompt_list = [
        # {k: v.format(**answer_dict) if isinstance(v, str) else v for k, v in prompt.items()} for prompt in prompt_list
        {k: v.format_map(CustomDefaultDict(**replace_dict)) if isinstance(v, str) else v for k, v in prompt.items()}
        for prompt in prompt_list
    ]
    return prompt_list


def load_constants(file_path, **kwargs):
    # This function loads some constant variables from a python file, used for changing prompts and hyper-params
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Import constants into global namespace
    for key, value in vars(module).items():
        if key.isupper():  # Only uppercase constants
            globals()[key] = value

    for key, value in kwargs.items():
        if key.isupper():
            globals()[key] = value
    return module


def preprocess_data(data, input_template=None, input_key: List[str] = ["problem"], apply_chat_template=None) -> str:
    needed_inputs = {k: data[k] for k in input_key}
    if apply_chat_template:
        # problem = data[input_key]
        if MESSAGE_TEMPLATE is None:
            # Use the default templaet
            # if isinstance(problem, str):
            # chat = [{"role": "user", "content": problem}]
            chat = {"role": "user", "content": data[input_key[0]]}
        else:
            chat = apply_prompt(MESSAGE_TEMPLATE, replace_dict=needed_inputs)
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        # prompt = data[input_key]
        if input_template:
            prompt = input_template.format_map(CustomDefaultDict(**needed_inputs))
    return prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template)
            prompt = prompt.lstrip(
                tokenizer.bos_token
            )  # removing leading bos token, will be added in tokenize_fn when making experience
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]


class PromptDatasetWithGT(PromptDataset):
    # NOTE: as of 01/21/25, this dataset class cannot support blending with other datasets due to the different __getitem__ signature
    def __init__(self, dataset, tokenizer, strategy, input_template=None) -> None:
        super().__init__(dataset, tokenizer, strategy, input_template)

        answer_key = getattr(self.strategy.args, "answer_key", None)
        assert answer_key is not None, "answer_key is required for PromptDatasetWithGTanswer"

        self.answers: List[str] = list(map(lambda x: x[answer_key], dataset))

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "gt_answer": self.answers[idx]}
