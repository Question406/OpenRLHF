# This file contains functions for verifiable reward.
from typing import List


def math_correctness_reward_fn(
    gt_answers: List[str],
    responses: List[str],
) -> List[int]:
    def assign_reward(gt_answer, response):
        try:
            gt_answer = float(gt_answer)
            response = float(response)
            return int(gt_answer == response)
        except ValueError:
            return 0.0

    rewards = [assign_reward(gt_answer, response) for gt_answer, response in zip(gt_answers, responses)]
    return rewards


def format_reward_fn(responses: List[str]) -> List[float]:
    def assign_reward(response):
        tags = [
            "<think>",
            "</think>",
            "<answer>",
            "</answer>",
        ]
        for tag in tags:
            if response.count(tag) != 1:
                return 0.0
        if not response.strip().startswith("<think>"):
            return 0.0

        # at least the response contains the desired tags
        left_think = response.find("<think>")
        right_think = response.find("</think>")
        left_answer = response.find("<answer>")
        right_answer = response.find("</answer>")
        return float((left_think < right_think) and (left_answer < right_answer) and (right_think < left_answer))

    rewards = [assign_reward(response) for response in responses]
    return rewards


def combined_reward(gt_answers, responses) -> List[float]:
    answer_reward = math_correctness_reward_fn(gt_answers, responses)
    format_rewards = format_reward_fn(responses)
    return [a + b for a, b in zip(answer_reward, format_rewards)]
