import json
from typing import List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from eval import eval_common_gen
import templates
import os
from dotenv import load_dotenv
from chat_templates import llama_template

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL = "meta-llama/Llama-3.2-3B-Instruct"
BATCH_SIZE = 48

# Load environment variables from .env file
load_dotenv()

# Get token from environment variables
hf_token = os.getenv("HUGGING_FACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGING_FACE_TOKEN not found in environment variables")

os.environ["HF_TOKEN"] = hf_token


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

    except Exception as e:
        print(f"Error getting chat template: {str(e)}")
        return None


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


def run_feedback(prompts: List[List[str]], responses: List[str], model, tokenizer):
    print("Running feedback...")

    # # print prompts
    # for i, prompt in enumerate(prompts):
    #     print(f"Prompt {i}:")
    #     print(prompt)
    #     print("-" * 100)
    # # print responses
    # for i, response in enumerate(responses):
    #     print(f"Response {i}:")
    #     print(response)
    #     print("-" * 100)

    # append each response as an assistant message to the list of prompts
    # each prompt is a list of messages, so we need to append the assistant message to each prompt
    for prompt, response in zip(prompts, responses):
        prompt.append({"role": "assistant", "content": f"{response}"})

    # print prompts
    # for i, prompt in enumerate(prompts):
    #     print(f"Prompt {i}:")
    #     print(prompt)
    #     print("-" * 100)

    # run the feedback template
    # iterate over each prompt and append the feedback template
    feedback_prompts = [templates.feedback_template(prompt) for prompt in prompts]

    # print prompts
    # for i, prompt in enumerate(feedback_prompts):
    #     print(f"Prompt {i}:")
    #     print(prompt)
    #     print("-" * 100)

    feedback_responses = batch_generate(feedback_prompts, model, tokenizer)

    # print each feedback response
    # for i, response in enumerate(feedback_responses):
    #     # print(f"Prompt {i}:")
    #     # print(prompts[i])
    #     # print("-" * 100)
    #     print(f"Feedback response {i}:")
    #     print(response)
    #     print("-" * 100)

    return feedback_responses


def run_refine(
    prompts: List[List[str]],
    responses: List[str],
    model,
    tokenizer,
    instructions: str = None,
):
    print("Running refine...")

    # append each response as an assistant message to the list of prompts
    for prompt, response in zip(prompts, responses):
        prompt.append({"role": "assistant", "content": f"{response}"})

    # run the refine template
    # iterate over each prompt and append the refine template
    refine_prompts = [
        templates.refine_template(prompt, instructions) for prompt in prompts
    ]
    refine_responses = batch_generate(refine_prompts, model, tokenizer)

    # print each refine response
    # for i, response in enumerate(refine_responses):
    #     # print(f"Prompt {i}:")
    #     # print(prompts[i])
    #     # print("-" * 100)
    #     print(f"Refine response {i}:")
    #     print(response)
    #     print("-" * 100)

    return refine_responses


# For each run, we want to first use the unmodified template,
# then use the modified template, finally we use 1 feedback + refine iteration.
# We will have multiple results:
# 1. Initial Unmodified template results
# 2. Initial Modified template results
# 3. Post-feedback + refinement ressults for both


def run_gsm(model, tokenizer):
    # load the gsm data
    with open("data/gsm.jsonl", "r") as f:
        gsm_data = [json.loads(line) for line in f.readlines()]

    gsm_data = gsm_data[:1]
    # use only input field
    gsm_data = [prompt["input"] for prompt in gsm_data]
    prompts_unmodified = [templates.gsm_template(prompt) for prompt in gsm_data]
    prompts_modified = [templates.gsm_template_modified(prompt) for prompt in gsm_data]

    # run the unmodified + modified templates
    responses_unmodified = batch_generate(prompts_unmodified, model, tokenizer)
    responses_modified = batch_generate(prompts_modified, model, tokenizer)

    # print each response
    print("Unmodified responses:")
    for i, response in enumerate(responses_unmodified):
        # prompt
        print(f"Prompt {i}:")
        print(prompts_unmodified[i])
        print("-" * 100)
        # response
        print(f"Response {i}:")
        print(response)
        print("-" * 100)

    print("Modified responses:")
    for i, response in enumerate(responses_modified):
        print(f"Prompt {i}:")
        print(prompts_modified[i])
        print("-" * 100)
        # response
        print(f"Response {i}:")
        print(response)
        print("-" * 100)


def run_common_gen(model, tokenizer):
    # load the common gen data
    with open("data/commongen_hard.jsonl", "r") as f:
        commongen_data = [json.loads(line) for line in f.readlines()]

    commongen_data = commongen_data[:40]
    prompts_unmodified = [
        templates.common_gen_template(prompt["concepts"]) for prompt in commongen_data
    ]
    prompts_modified = [
        templates.common_gen_template_modified(prompt["concepts"])
        for prompt in commongen_data
    ]

    # run the unmodified + modified templates
    responses_unmodified = batch_generate(prompts_unmodified, model, tokenizer)
    responses_modified = batch_generate(prompts_modified, model, tokenizer)

    # print each response
    # print("Unmodified responses:")
    # for i, response in enumerate(responses_unmodified):
    #     # print(f"Prompt {i}:")
    #     # print(prompts_unmodified[i])
    #     # print("-" * 100)
    #     print(f"Response {i}:")
    #     print(response)
    #     print("-" * 100)

    # print("Modified responses:")
    # for i, response in enumerate(responses_modified):
    #     # print(f"Prompt {i}:")
    #     # print(prompts_modified[i])
    #     # print("-" * 100)
    #     print(f"Response {i}:")
    #     print(response)
    #     print("-" * 100)

    # run feedback + refine for unmodified
    feedback_unmodified = run_feedback(
        prompts_unmodified, responses_unmodified, model, tokenizer
    )
    refine_unmodified = run_refine(
        prompts_unmodified, feedback_unmodified, model, tokenizer
    )

    # run feedback + refine for modified
    feedback_modified = run_feedback(
        prompts_modified, responses_modified, model, tokenizer
    )
    refine_modified = run_refine(prompts_modified, feedback_modified, model, tokenizer)

    # eval the pre-refine responses
    print("Pre-refine:")
    # Calculate average scores for unmodified pre-refine
    unmod_pre_scores = [
        eval_common_gen(commongen_data[i]["concepts"], response)
        for i, response in enumerate(responses_unmodified)
    ]
    print(f"Unmodified avg: {sum(unmod_pre_scores)/len(unmod_pre_scores):.2f}")

    # Calculate average scores for modified pre-refine
    mod_pre_scores = [
        eval_common_gen(commongen_data[i]["concepts"], response)
        for i, response in enumerate(responses_modified)
    ]
    print(f"Modified avg: {sum(mod_pre_scores)/len(mod_pre_scores):.2f}")

    print("\nPost-refine:")
    # Calculate average scores for unmodified post-refine
    unmod_post_scores = [
        eval_common_gen(commongen_data[i]["concepts"], response)
        for i, response in enumerate(refine_unmodified)
    ]
    print(f"Unmodified avg: {sum(unmod_post_scores)/len(unmod_post_scores):.2f}")

    # Calculate average scores for modified post-refine
    mod_post_scores = [
        eval_common_gen(commongen_data[i]["concepts"], response)
        for i, response in enumerate(refine_modified)
    ]
    print(f"Modified avg: {sum(mod_post_scores)/len(mod_post_scores):.2f}")


def run(model_name: str):
    # initialize the model and tokenizer
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Instruct variants of models should already have a default chat template
    chat_template = get_chat_template(tokenizer, model_name)
    tokenizer.chat_template = chat_template

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # --- Run the tasks ---

    # run_gsm(model, tokenizer)
    run_common_gen(model, tokenizer)


if __name__ == "__main__":
    run(MODEL)
