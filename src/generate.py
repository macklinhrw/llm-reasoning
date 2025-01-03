from typing import List
from tqdm import tqdm
import torch

BATCH_SIZE = 48


def batch_generate(prompts: List[List[str]], model, tokenizer):
    # do batch inference
    responses = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
        batch_messages = prompts[i : i + BATCH_SIZE]

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
                **batch_inputs, max_new_tokens=1024, temperature=0.7
            )

        # decode the outputs
        for _, (input_ids, output_ids) in enumerate(
            zip(batch_inputs.input_ids, generated_ids)
        ):
            generated_part = output_ids[len(input_ids) :]
            response = tokenizer.decode(generated_part, skip_special_tokens=True)
            responses.append(response)

    return responses
