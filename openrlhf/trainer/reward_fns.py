# This file contains functions for verifiable reward.
from typing import List


def math_correctness_reward():
    pass


def format_reward(responses: List[str]) -> List[int]:
    def assign_reward(response):
        tags = [
            "<think>",
            "</think>",
            "<answer>",
            "</answer>",
        ]
        for tag in tags:
            if response.count(tag) != 1:
                return 0
        # at least the response contains the desired tags
        left_think = response.find("<think>")
        right_think = response.find("</think>")
        left_answer = response.find("<answer>")
        right_answer = response.find("</answer>")
        return int((left_think < right_think) and (left_answer < right_answer) and (right_think < left_answer))

    rewards = [assign_reward(response) for response in responses]
    return rewards


def combined_reward(gt_answers, responses):
    pass
