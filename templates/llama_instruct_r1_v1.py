USER_REQUEST = """Below is a math question. Provide your answer in following format. The answer is divided into two parts:
1. The reasoning process, enclosed within <think> and </think> tags.
2. The answer enclosed within <answer> and </answer> tags. You should put the final result within the \\boxed{{}}.
Now answer following question: {problem}
"""
USER_REQUEST = """{problem}"""

MESSAGE_TEMPLATE = [
    {
        "role": "system",
        "content": "You're a math expert answering questions given by a user. You should first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer should be enclosed within <think> and </think> tags and <answer> and </answer> tags respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.",
    },
    {
        "role": "user",
        "content": "{problem}",
    },
]
