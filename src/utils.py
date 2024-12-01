import re


def format_few_shot_examples(examples):
    messages = []
    for example in examples:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": format_question_prompt(example["question"]),
                },
                {"role": "assistant", "content": example["answer"]},
            ]
        )

    return messages


def format_question_prompt(question):
    return f'Given the following problem, reason and give a final answer to the problem.\nProblem: {question}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.'


# https://github.com/allenai/olmes/blob/main/oe_eval/tasks/utils.py
# def extract_answer(
#     continuation: str, answer_regexes, prefix_regexes=None, use_last_raw_match=True
# ):
#     # Search continuation for any prefix regex, extract first answer following it using answer regexes
#     if prefix_regexes:
#         for prefix_regex in prefix_regexes:
#             match = re.search(prefix_regex, continuation)
#             if match:
#                 rest = continuation[match.end() :].strip()
#                 # search for answer at the start of the rest:
#                 answer_match = re.findall("^(" + "|".join(answer_regexes) + ")", rest)
#                 if answer_match:
#                     return answer_match[0]
#     for answer_regex in answer_regexes:
#         ans_match = re.findall(answer_regex, continuation)
#         if ans_match:
#             if use_last_raw_match:
#                 return ans_match[-1]
#             else:
#                 return ans_match[0]
#     return None


def parse_number(text):
    """Parse a number string, handling commas."""
    try:
        return float(text.replace(",", "").strip())
    except Exception as e:
        print(f"\nError parsing number.")
        print(f"Raw text: {text}")
        raise e


def extract_answer(response):
    """Extract the final numerical answer from the response, handling commas."""
    try:
        # Handle case where response is a list
        if isinstance(response, list):
            response = response[0]

        response = response.replace(",", "")
        response = response.replace("$", "")
        # this regex seems fine for GSM8K
        numbers = re.findall(r"-?\d*\.?\d+", response)
        return float(numbers[-1]) if numbers else None
    except Exception as e:
        print(f"Error extracting answer from: {response}")
        return None
