from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import re
import numpy as np
from tqdm import tqdm
import os
from dotenv import load_dotenv
from prompts import (
    gsm8k_few_shot_prompt,
    gsm8k_zero_shot_prompt,
    llama3_2_gsm8k_few_shot_examples,
    llama3_2_gsm8k_instruction_prompt,
    full_gsm8k_zero_shot_prompt,
)
import datetime
import json
from utils import (
    format_few_shot_examples,
    format_question_prompt,
    extract_answer,
    parse_number,
    get_gsm8k_answers,
)
from chat_templates import llama_template
from test_evals import test_evals, compare_evals, evaluate_model_response
from generate import batch_generate, k_shot_generate, self_consistency_generate

# Load environment variables from .env file
load_dotenv()

# Get token from environment variables
hf_token = os.getenv("HUGGING_FACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGING_FACE_TOKEN not found in environment variables")

os.environ["HF_TOKEN"] = hf_token


# Claude-3.5-sonnet generated code -- currently working on refactoring


def get_chat_template(tokenizer, model_name):
    # Some chat templates we might want to override, which are
    # at the top. Other templates will be the default from the tokenizer.
    # Otherwise we provide our own.
    try:

        # Overwrite llama template
        if "llama" in model_name.lower():
            return llama_template

        if tokenizer.chat_template is not None:
            return tokenizer.chat_template

        # Try to get the default template from the model's config
        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, "chat_template"):
            return config.chat_template

    # These need need to be checked against official templates (these are generated by Claude)

    #         # Check if it's a known model with a specific template
    #         if "mistral" in model_name.lower():
    #             return "<s>[INST] {{ if system }} {{ system }} {{ end }}{{ query }} [/INST] {{ response }} </s>"
    #         elif "llama" in model_name.lower():
    #             return "[INST] {% if system %} {{ system }} {% endif %} {{ user }} [/INST] {{ assistant }}"
    #         elif "phi" in model_name.lower():
    #             return "Instruct: {{ user }}\nOutput: {{ assistant }}"
    #         else:
    #             # Fallback to a generic template
    #             return """{% for message in messages %}{% if message['role'] == 'system' %}System: {{ message['content'] }}
    # {% elif message['role'] == 'user' %}User: {{ message['content'] }}
    # {% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}
    # {% endif %}{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}"""

    except Exception as e:
        print(f"Error getting chat template: {str(e)}")
        return None


def batch_evaluate_gsm8k(
    model_name,
    model,
    tokenizer,
    batch_size=8,
    num_samples=None,
    instruction_prompt=None,
    examples=None,
):
    dataset = load_dataset("gsm8k", "main")["test"]

    if num_samples:
        dataset = dataset.select(range(num_samples))

    correct = 0
    total = 0

    print(f"Starting evaluation with batch size: {batch_size}")

    results_log = []

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch_slice = slice(i, min(i + batch_size, len(dataset)))
        batch_data = dataset[batch_slice]

        questions = batch_data["question"]
        correct_answers = [
            parse_number(answer.split("####")[-1].strip())
            for answer in batch_data["answer"]
        ]

        if isinstance(questions, str):
            questions = [questions]
            correct_answers = [correct_answers]

        batch_messages = [
            [
                {"role": "user", "content": format_question_prompt(question)},
            ]
            for question in questions
        ]

        if examples:
            batch_messages = [
                # [
                #     # few shot examples
                #     # only if examples are provided
                #     *format_few_shot_examples(examples),
                # ]
                # for question in questions
                format_few_shot_examples(examples) + batch_messages[i]
                for i in range(len(batch_messages))
            ]

        if instruction_prompt:
            # prepend instruction prompt to list
            batch_messages = [
                [{"role": "system", "content": instruction_prompt}] + batch_messages[i]
                for i in range(len(batch_messages))
            ]

        batch_texts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in batch_messages
        ]

        # gives us a list of tensors, one for each question in the batch
        batch_inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            # Set random seed for reproducibility
            torch.manual_seed(42)
            # we can then pass this batch of tensors to the model
            # and it will generate in parallel for each question (batch inference)
            generated_ids = model.generate(
                **batch_inputs,
                max_new_tokens=1024,
                do_sample=False,  # Greedy decoding
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                # top_k=0,
                # top_p=0,
                # temperature=0.0,
            )

        for j, (input_ids, output_ids) in enumerate(
            zip(batch_inputs.input_ids, generated_ids)
        ):
            generated_part = output_ids[len(input_ids) :]
            response = tokenizer.decode(generated_part, skip_special_tokens=True)

            predicted_answer, is_correct = evaluate_model_response(
                response, correct_answers[j]
            )

            results_log.append(
                {
                    "question": questions[j],
                    "response": response,
                    "predicted": predicted_answer,
                    "correct": correct_answers[j],
                    "prompt": batch_texts[j],
                    "is_correct": is_correct,
                }
            )

            if is_correct:
                correct += 1
            total += 1

        # Print progress after each batch
        print(
            f"Progress: {total}/{len(dataset)} - Current Accuracy: {(correct/total)*100:.2f}%"
        )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    model_name_short = model_name.split("/")[-1]
    results_file = f"results/evaluation_results-{model_name_short}-{timestamp}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results_log, f, indent=2, default=str)

    final_accuracy = (correct / total) * 100
    return final_accuracy


def run_evaluation(
    model,
    tokenizer,
    generate_fn,
    batch_size=8,
    instruction_prompt=None,
    examples=None,
    num_samples=None,
    generate_kwargs={},
):
    """
    Evaluates model accuracy on gsm8k dataset.
    Num samples is optional, if None, all samples are used.
    generate_kwargs allows passing additional arguments to the generate function
    """
    try:
        # get gsm8k dataset
        dataset = load_dataset("gsm8k", "main")["test"]

        if num_samples:
            dataset = dataset.select(range(num_samples))

        correct = 0
        total = 0

        # Create results dictionary with metadata
        results_data = {
            "parameters": {
                "model_name": model.config._name_or_path,
                "generation_method": generate_fn.__name__,
                "batch_size": batch_size,
                "num_samples": num_samples,
                "has_instruction_prompt": instruction_prompt is not None,
                "has_examples": examples is not None,
                "generate_kwargs": generate_kwargs,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"),
            },
            "results": [],  # This will store the individual evaluation results
        }

        for i in tqdm(range(0, len(dataset), batch_size)):
            batch_slice = slice(i, min(i + batch_size, len(dataset)))
            batch_data = dataset[batch_slice]

            questions = batch_data["question"]
            correct_answers = [
                parse_number(answer.split("####")[-1].strip())
                for answer in batch_data["answer"]
            ]

            batch_messages = [
                [
                    {"role": "user", "content": format_question_prompt(question)},
                ]
                for question in questions
            ]

            # Examples and instruction prompt
            if examples:
                batch_messages = [
                    format_few_shot_examples(examples) + batch_messages[i]
                    for i in range(len(batch_messages))
                ]

            if instruction_prompt:
                batch_messages = [
                    [{"role": "system", "content": instruction_prompt}]
                    + batch_messages[i]
                    for i in range(len(batch_messages))
                ]

            # 1. batch generate for current batch
            responses = generate_fn(batch_messages, model, tokenizer, **generate_kwargs)
            # 2. run evals and get accuracy
            for j, (response_data, correct_answer) in enumerate(
                zip(responses, correct_answers)
            ):
                # Handle different response types
                if isinstance(response_data, dict):
                    # Self-consistency case
                    response = response_data["majority"]
                    all_responses = response_data["all_responses"]
                    extra_data = {
                        "all_responses": all_responses,
                        "majority_answer": response_data["majority_answer"],
                        "answer_counts": response_data["answer_counts"],
                    }
                elif isinstance(response_data, list):
                    # K-shot case
                    all_responses = response_data
                    response = response_data[0]  # Use first response for logging
                    extra_data = {"all_responses": all_responses}
                else:
                    # Single response case
                    response = response_data
                    all_responses = [response_data]
                    extra_data = {}

                # Check correctness based on generation method
                if isinstance(response_data, dict):
                    # Self-consistency case - only check majority answer
                    response = response_data["majority"]
                    predicted_answer, is_correct = evaluate_model_response(
                        response, correct_answer
                    )
                    predicted_answers = [predicted_answer]
                else:
                    # K-shot case - check if any response is correct
                    is_correct = False
                    predicted_answers = []
                    for single_response in all_responses:
                        pred_answer, single_correct = evaluate_model_response(
                            single_response, correct_answer
                        )
                        predicted_answers.append(pred_answer)
                        if single_correct:
                            is_correct = True
                            break

                if is_correct:
                    correct += 1
                total += 1

                # Update results_data["results"]
                results_data["results"].append(
                    {
                        "question": questions[j],
                        "response": response,  # Primary response for display
                        "predicted": predicted_answers,  # All predicted answers
                        "correct": correct_answer,
                        "prompt": batch_messages[j],
                        "is_correct": is_correct,
                        **extra_data,
                    }
                )

                # Print progress after each example
                if total % 10 == 0:  # Print every 10 examples
                    print(
                        f"Progress: {total}/{len(dataset)} - Current Accuracy: {(correct/total)*100:.2f}%"
                    )

        # Save results
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        model_name_short = model.config._name_or_path.split("/")[-1]
        generate_fn_name = generate_fn.__name__
        results_file = f"results/evaluation_results-{model_name_short}-{generate_fn_name}-{timestamp}.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

        # Add final accuracy to parameters
        results_data["parameters"]["final_accuracy"] = (correct / total) * 100

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        return (correct / total) * 100

    except Exception as e:
        print(f"Evaluation failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def test_evals(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    # Instruct variants of models should already have a default chat template
    chat_template = get_chat_template(tokenizer, model_name)
    tokenizer.chat_template = chat_template

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compare_evals(tokenizer)


def load_model_and_tokenizer(model_name):
    """Load model and tokenizer with standard configuration"""
    # Enable Flash Attention
    has_flash_attn = True

    # Configure model with Flash Attention
    config = AutoConfig.from_pretrained(model_name)
    if has_flash_attn:
        if hasattr(config, "use_flash_attention_2"):
            config.use_flash_attention_2 = True
        if hasattr(config, "attention_mode"):
            config.attention_mode = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    # Instruct variants of models should already have a default chat template
    chat_template = get_chat_template(tokenizer, model_name)
    tokenizer.chat_template = chat_template

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


models = [
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

if __name__ == "__main__":
    model_name = models[2]
    model, tokenizer = load_model_and_tokenizer(model_name)

    # accuracy = run_evaluation(
    #     model=model,
    #     tokenizer=tokenizer,
    #     generate_fn=batch_generate,
    #     batch_size=64,
    #     instruction_prompt=full_gsm8k_zero_shot_prompt,
    #     generate_kwargs={
    #         "batch_size": 64,
    #     },
    # )
    # print(f"Accuracy (regular): {accuracy:.2f}%")

    accuracy = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        generate_fn=k_shot_generate,
        batch_size=64,
        num_samples=None,
        # instruction_prompt=full_gsm8k_zero_shot_prompt,
        generate_kwargs={
            "k": 3,
            "batch_size": 64,
            "temperature": 0.7,
        },
    )
    print(f"Accuracy (k-shot): {accuracy:.2f}%")

    accuracy = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        generate_fn=self_consistency_generate,
        batch_size=64,
        num_samples=None,
        # instruction_prompt=full_gsm8k_zero_shot_prompt,
        generate_kwargs={
            "k": 3,
            "batch_size": 64,
            "extract_fn": extract_answer,
            "temperature": 0.7,
        },
    )
    print(f"Accuracy (self-consistency): {accuracy:.2f}%")

    # test_evals(model_name)
