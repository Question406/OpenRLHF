USER_REQUEST = """Below is a math question. Provide your answer in following format. The answer is divided into two parts:
1. The reasoning process, enclosed within <think> and </think> tags.
2. The answer enclosed within <answer> and </answer> tags. You should put the final result within the \\boxed{{}}.
Now answer following question: {problem}
"""

MESSAGE_TEMPLATE = [
    {
        "role": "system",
        "content": "You're a math expert answering questions according to the requirement given by the user.",
    },
    {
        "role": "user",
        "content": USER_REQUEST,
    },
]
