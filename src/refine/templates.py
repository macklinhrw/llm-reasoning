import json
from typing import List


def gsm_template(question: str):
    with open("data/gsm_init.txt", "r") as examples:
        examples = examples.read()

    ## use chat template format
    messages = []
    messages.append(
        {
            "role": "system",
            "content": "Solve the problem using Python according to the examples and provide the final answer as a function named `solution()`.",
        }
    )
    # multi-turn dialogue for examples
    examples_split = examples.split("\n\n\n")
    for example in examples_split:
        if example.strip() == "":
            continue

        qa_split = example.split("# solution using Python:\n")
        question_qa = qa_split[0].replace("# Q: ", "").strip()
        answer_qa = qa_split[1].strip()
        messages.append(
            {"role": "user", "content": f"{question_qa}"},
        )
        messages.append(
            {"role": "assistant", "content": f"{answer_qa}"},
        )
    messages.append(
        {"role": "user", "content": f"{question}"},
    )
    return messages


def gsm_template_modified(question: str):
    with open("data/gsm_init.txt", "r") as examples:
        examples = examples.read()

    ## use chat template format
    messages = []
    messages.append(
        {
            "role": "system",
            "content": "Using Python plan your solution to the problem step by step in comments and explain your reasoning, then provide the final answer as a function named `solution()`.",
        }
    )
    # # multi-turn dialogue for examples
    # examples_split = examples.split("\n\n\n")
    # for example in examples_split:
    #     if example.strip() == "":
    #         continue

    #     qa_split = example.split("# solution using Python:\n")
    #     question_qa = qa_split[0].replace("# Q: ", "").strip()
    #     answer_qa = qa_split[1].strip()
    #     messages.append(
    #         {"role": "user", "content": f"{question_qa}"},
    #     )
    #     messages.append(
    #         {"role": "assistant", "content": f"{answer_qa}"},
    #     )
    messages.append(
        {"role": "user", "content": f"{question}"},
    )
    return messages


def common_gen_template(concepts: List[str]):
    with open("data/commongen_init.jsonl", "r") as examples_file:
        # examples = ""
        examples_raw = examples_file.read()
        examples_json = [
            json.loads(example)
            for example in examples_raw.split("\n")
            if example.strip()
        ]

    ## use chat template format
    messages = []
    # prompt = "Write a sentence using the concepts."

    for example in examples_json:
        messages.append(
            {"role": "user", "content": f"Concepts: {example['concepts']}"},
        )
        messages.append(
            {"role": "assistant", "content": f"Sentence: {example['answer']}"},
        )

    messages.append(
        {"role": "user", "content": f"Concepts: {concepts}"},
    )
    return messages


def common_gen_template_modified(concepts: List[str]):
    with open("data/commongen_init.jsonl", "r") as examples_file:
        # examples = ""
        examples_raw = examples_file.read()
        examples_json = [
            json.loads(example)
            for example in examples_raw.split("\n")
            if example.strip()
        ]

    ## use chat template format
    messages = []
    modify_prompt = (
        "Write a reasonable paragraph that includes *ALL* of the above concepts."
    )

    for example in examples_json:
        messages.append(
            {"role": "user", "content": f"Concepts: {example['concepts']}"},
        )
        messages.append(
            {"role": "assistant", "content": f"Sentence: {example['answer']}"},
        )

    messages.append(
        {"role": "user", "content": f"Concepts: {concepts}\n{modify_prompt}\n"},
    )
    return messages


def feedback_template(messages: List[str], prompt: str = None):
    feedback_prompt = "Review your previous answer and find problems with your answer."
    if prompt:
        feedback_prompt = prompt
    messages.append({"role": "user", "content": f"{feedback_prompt}"})
    return messages


def refine_template(messages: List[str], instructions: str, prompt: str = None):
    """
    Instructions are for answer formatting.
    """
    refine_prompt = (
        f"Based on the problems you found, improve your answer. {instructions}"
    ).strip()
    if prompt:
        refine_prompt = f"{prompt}\n{instructions}"
    messages.append({"role": "user", "content": f"{refine_prompt}"})
    return messages
