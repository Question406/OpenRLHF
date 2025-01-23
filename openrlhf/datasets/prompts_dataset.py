from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List


class CustomDefaultDict(dict):
    def __missing__(self, key):
        return f"{{{key}}}"  # Keep the placeholder as-is


def preprocess_data(data, input_template=None, input_key="input", apply_chat_template=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format_map(CustomDefaultDict(problem=prompt))
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
