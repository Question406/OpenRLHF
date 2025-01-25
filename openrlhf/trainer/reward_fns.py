# This file contains functions for verifiable reward.
import re
from typing import List
from openrlhf.trainer.eval_utils import math_equal
from openrlhf.trainer.answer_extraction import extract_answer
from openrlhf.utils.registry_utils import Registry

REWARD_REGISTOR = Registry()


def math_correctness_reward_fn(
    gt_answers: List[str],
    responses: List[str],
) -> List[int]:
    def assign_reward(gt_answer, response):
        return float(math_equal(extract_answer(response), gt_answer))

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
        if not response.strip().endswith("</answer>"):
            return 0.0

        # at least the response contains the desired tags
        left_think = response.find("<think>")
        right_think = response.find("</think>")
        left_answer = response.find("<answer>")
        right_answer = response.find("</answer>")
        return float((left_think < right_think) and (left_answer < right_answer) and (right_think < left_answer))

    rewards = [assign_reward(response) for response in responses]
    return rewards


def combined_reward(gt_answers: List[str], responses: List[str]) -> List[float]:
    answer_reward = math_correctness_reward_fn(gt_answers, responses)
    format_rewards = format_reward_fn(responses)
    return [a + b for a, b in zip(answer_reward, format_rewards)]


# Countdown dataset reward, copied from https://github.com/Jiayi-Pan/TinyZero.git
def countdown_reward_fn(
    gt_answers: List[str],
    responses: List[str],
):
    def extract_solution(solution_str):
        """Extract the equation from the solution string."""
        # Remove everything before the first "Assistant:"
        if "Assistant:" in solution_str:
            solution_str = solution_str.split("Assistant:", 1)[1]
        elif "<|im_start|>assistant" in solution_str:
            solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
        else:
            return None
        solution_str = solution_str.split("\n")[-1]

        answer_pattern = r"<answer>(.*?)</answer>"
        match = re.finditer(answer_pattern, solution_str)
        matches = list(match)
        if matches:
            final_answer = matches[-1].group(1).strip()
        else:
            final_answer = None
        return final_answer

    def validate_equation(equation_str, available_numbers):
        """Validate that equation only uses available numbers and each number once."""
        try:
            # Extract all numbers from the equation
            numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

            # Check if all numbers in equation are available
            available_numbers = sorted(available_numbers)
            numbers_in_eq = sorted(numbers_in_eq)

            # Each number should be used exactly once
            return numbers_in_eq == available_numbers
        except Exception:
            return False

    def evaluate_equation(equation_str):
        """Safely evaluate the arithmetic equation using eval() with precautions."""
        try:
            # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
            allowed_pattern = r"^[\d+\-*/().\s]+$"
            if not re.match(allowed_pattern, equation_str):
                raise ValueError("Invalid characters in equation.")

            # Evaluate the equation with restricted globals and locals
            result = eval(equation_str, {"__builtins__": None}, {})
            return result
        except Exception:
            return None

    def assign_reward(gt_answer, response):
        numbers, target = gt_answer.split("=")
        numbers = eval(numbers)
        target = eval(target)
        format_score = 0.1
        score = 1.0

        equation = extract_solution(solution_str=response)
        # Validate equation uses correct numbers
        if not validate_equation(equation, numbers):
            return format_score

        # Evaluate equation
        try:
            result = evaluate_equation(equation)
            if result is None:
                return format_score

            if abs(result - target) < 1e-5:  # Account for floating point precision
                return score
            else:
                return format_score
        except Exception:
            return format_score

    rewards = [assign_reward(gt_answer, response) for gt_answer, response in zip(gt_answers, responses)]
    return rewards


REWARD_REGISTOR.register("countdown", countdown_reward_fn)
REWARD_REGISTOR.register("math_correctness", math_correctness_reward_fn)
REWARD_REGISTOR.register("format", format_reward_fn)
REWARD_REGISTOR.register("math", combined_reward)
REWARD_REGISTOR.register("gsm8k", combined_reward)
