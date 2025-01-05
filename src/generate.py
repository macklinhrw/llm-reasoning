from collections import Counter
from typing import List
from tqdm import tqdm
import torch


def batch_generate(
    prompts: List[List[str]],
    model,
    tokenizer,
    batch_size: int = 10,
    temperature: float = 0.7,
):
    # do batch inference
    responses = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_messages = prompts[i : i + batch_size]

        batch_texts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in batch_messages
        ]

        batch_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(
            model.device
        )

        with torch.no_grad():
            generated_ids = model.generate(
                **batch_inputs, max_new_tokens=1024, temperature=temperature
            )

        # decode the outputs
        for _, (input_ids, output_ids) in enumerate(
            zip(batch_inputs.input_ids, generated_ids)
        ):
            generated_part = output_ids[len(input_ids) :]
            response = tokenizer.decode(generated_part, skip_special_tokens=True)
            responses.append(response)

    return responses


def k_shot_generate(
    prompts: List[List[str]],
    model,
    tokenizer,
    k: int = 10,
    batch_size: int = 10,
    temperature: float = 0.7,
):
    # First, create k copies of each prompt consecutively
    k_shot_prompts = []
    for prompt in prompts:
        k_shot_prompts.extend([prompt] * k)  # Add k copies of the same prompt together

    responses = batch_generate(
        k_shot_prompts, model, tokenizer, batch_size, temperature
    )

    # Group the responses by original prompt
    k_shot_responses = []
    for i in range(0, len(responses), k):
        group = responses[i : i + k]
        k_shot_responses.append(group)

    return k_shot_responses


# Assuming eval_fn takes (predicted answer, correct answer)
def self_consistency_generate(
    prompts: List[List[str]],
    model,
    tokenizer,
    extract_fn=None,
    k: int = 10,
    batch_size: int = 10,
    temperature: float = 0.7,
):
    responses = k_shot_generate(prompts, model, tokenizer, k, batch_size, temperature)

    if extract_fn is None:
        raise ValueError("extract_fn must be provided for self-consistency generation")

    majority_votes = []
    all_responses = []  # To store all responses for each prompt
    for i in range(len(responses)):
        k_shot_responses = responses[i]
        # First get all extracted answers with their indices
        answers_with_idx = [
            (extract_fn(resp), idx) for idx, resp in enumerate(k_shot_responses)
        ]

        # Count just the answers
        answer_counts = Counter(ans for ans, _ in answers_with_idx)
        majority_answer = answer_counts.most_common(1)[0][0]

        # Find index of a response that gave the majority answer
        majority_idx = next(
            idx for ans, idx in answers_with_idx if ans == majority_answer
        )

        # Use that index to get the full response
        majority_response = k_shot_responses[majority_idx]
        majority_votes.append(majority_response)

        # Store all responses for this prompt
        all_responses.append(
            {
                "majority": majority_response,
                "all_responses": k_shot_responses,
                "majority_answer": majority_answer,
                "answer_counts": dict(answer_counts),
            }
        )

    return all_responses
