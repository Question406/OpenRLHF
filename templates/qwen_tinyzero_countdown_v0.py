USER_REQUEST = """

"""

MESSAGE_TEMPLATE = [
    {
        "role": "system",
        "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.",
    },
    {
        "role": "user",
        "content": "Using the numbers {problem}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.",
    },
]
