from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import re
import numpy as np
from tqdm import tqdm
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get token from environment variables
hf_token = os.getenv("HUGGING_FACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGING_FACE_TOKEN not found in environment variables")


# Claude-3.5-sonnet generated code -- currently working on refactoring
def get_default_chat_template(model_name):
    try:
        # Try to get the default template from the model's config
        config = AutoConfig.from_pretrained(model_name)
        if hasattr(config, "chat_template"):
            return config.chat_template

        # Check if it's a known model with a specific template
        if "mistral" in model_name.lower():
            return "<s>[INST] {{ if system }} {{ system }} {{ end }}{{ query }} [/INST] {{ response }} </s>"
        elif "llama-2" in model_name.lower():
            return "[INST] {% if system %} {{ system }} {% endif %} {{ user }} [/INST] {{ assistant }}"
        elif "phi" in model_name.lower():
            return "Instruct: {{ user }}\nOutput: {{ assistant }}"
        else:
            # Fallback to a generic template
            return """{% for message in messages %}{% if message['role'] == 'system' %}System: {{ message['content'] }}
{% elif message['role'] == 'user' %}User: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}
{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}"""

    except Exception as e:
        print(f"Error getting chat template: {str(e)}")
        return None


def parse_number(text):
    """Parse a number string, handling commas."""
    try:
        return float(text.replace(",", "").strip())
    except Exception as e:
        print(f"Error parsing number: {text}")
        raise e


def extract_answer(response):
    """Extract the final numerical answer from the response, handling commas."""
    try:
        response = response.replace(",", "")
        numbers = re.findall(r"-?\d*\.?\d+", response)
        return float(numbers[-1]) if numbers else None
    except Exception as e:
        print(f"Error extracting answer from: {response}")
        return None


def find_optimal_batch_size(model, tokenizer, max_batch_size=64, initial_batch_size=32):
    print("Finding optimal batch size...")

    sample_messages = [
        {
            "role": "system",
            "content": "You are a helpful math assistant. Solve the problem step by step.",
        },
        {"role": "user", "content": "What is 2+2?"},
    ]
    sample_text = tokenizer.apply_chat_template(
        sample_messages, tokenize=False, add_generation_prompt=True
    )

    current_batch_size = initial_batch_size

    while current_batch_size <= max_batch_size:
        try:
            print(f"Testing batch size: {current_batch_size}")
            batch_texts = [sample_text] * current_batch_size

            batch_inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True
            ).to(model.device)

            with torch.no_grad():
                _ = model.generate(
                    **batch_inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                )

            if current_batch_size == max_batch_size:
                break
            current_batch_size += 8

        except torch.cuda.OutOfMemoryError:
            current_batch_size -= 8
            break

    print(f"Optimal batch size found: {current_batch_size}")
    return current_batch_size


def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU Memory: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


def batch_evaluate_gsm8k(model, tokenizer, batch_size=8, num_samples=None):
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
                {
                    "role": "system",
                    "content": "You are a helpful math assistant. Solve the problem step by step and provide the final answer as a number.",
                },
                {"role": "user", "content": question},
            ]
            for question in questions
        ]

        batch_texts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in batch_messages
        ]

        batch_inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **batch_inputs,
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, (input_ids, output_ids) in enumerate(
            zip(batch_inputs.input_ids, generated_ids)
        ):
            generated_part = output_ids[len(input_ids) :]
            response = tokenizer.decode(generated_part, skip_special_tokens=True)

            predicted_answer = extract_answer(response)

            results_log.append(
                {
                    "question": questions[j],
                    "response": response,
                    "predicted": predicted_answer,
                    "correct": correct_answers[j],
                }
            )

            if predicted_answer is not None:
                if abs(predicted_answer - correct_answers[j]) < 1e-6:
                    correct += 1

            total += 1

        if (i + batch_size) % (batch_size * 5) == 0:
            print(
                f"Progress: {total}/{len(dataset)} - Current Accuracy: {(correct/total)*100:.2f}%"
            )

    import json

    with open("evaluation_results.json", "w") as f:
        json.dump(results_log, f, indent=2, default=str)

    final_accuracy = (correct / total) * 100
    return final_accuracy


def run_evaluation(model_name, num_samples=None):
    # Enable Flash Attention
    has_flash_attn = True

    # Configure model with Flash Attention
    config = AutoConfig.from_pretrained(model_name)
    if has_flash_attn:
        if hasattr(config, "use_flash_attention_2"):
            config.use_flash_attention_2 = True
        if hasattr(config, "attention_mode"):
            config.attention_mode = "flash_attention_2"

    # Load model with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name, config=config, torch_dtype=torch.float16, device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    chat_template = get_default_chat_template(model_name)
    if chat_template:
        tokenizer.chat_template = chat_template
    else:
        print("Warning: Using fallback chat template")
        tokenizer.chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}System: {{ message['content'] }}
{% elif message['role'] == 'user' %}User: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}
{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}"""

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Find optimal batch size
    batch_size = find_optimal_batch_size(model, tokenizer)

    # Print initial memory usage
    print_gpu_memory()

    try:
        accuracy = batch_evaluate_gsm8k(
            model, tokenizer, batch_size=batch_size, num_samples=num_samples
        )
        print(f"\nFinal Accuracy: {accuracy:.2f}%")
        return accuracy

    except Exception as e:
        print(f"Evaluation failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


models = ["meta-llama/Llama-3.2-3B", "Qwen/Qwen2.5-7B-Instruct"]

if __name__ == "__main__":
    model_name = models[0]

    accuracy = run_evaluation(
        model_name=model_name, num_samples=100  # Set to None for full dataset
    )
