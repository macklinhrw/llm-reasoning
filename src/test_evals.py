from datasets import load_dataset
from tqdm import tqdm
import os
import datetime
import json
from utils import (
    format_few_shot_examples,
    format_question_prompt,
    extract_answer,
    parse_number,
)
from chat_templates import llama_template
from prompts import (
    gsm8k_few_shot_prompt,
    gsm8k_zero_shot_prompt,
    llama3_2_gsm8k_few_shot_examples,
    llama3_2_gsm8k_instruction_prompt,
)


def evaluate_model_response(response, correct_answer):
    """
    Evaluate a model's response against the correct answer.

    Args:
        response (str): The complete text response from the model
        correct_answer (float): The expected numerical answer

    Returns:
        tuple: (predicted_answer, is_correct) where:
            - predicted_answer (float|None): The extracted numerical answer or None if parsing failed
            - is_correct (bool): Whether the answer matches within tolerance
    """
    predicted_answer = extract_answer(response)
    is_correct = evaluate_result(predicted_answer, correct_answer)
    return predicted_answer, is_correct


def evaluate_result(predicted_answer, correct_answer):
    """Evaluate if a predicted answer matches the correct answer within tolerance."""
    if predicted_answer is not None:
        return abs(predicted_answer - correct_answer) < 1e-6
    return False


def test_evals():
    """
    calculate the number of correct predictions on the gsm8k dataset
    """
    llama_evals = load_dataset(
        "meta-llama/Llama-3.2-1B-Instruct-evals",
        name="Llama-3.2-1B-Instruct-evals__gsm8k__details",
        split="latest",
    )

    total = len(llama_evals)
    parsing_matches = 0
    comparison_results = []

    for item in tqdm(llama_evals):
        try:
            # Print context when processing each item
            prediction_text = item["output_prediction_text"][0]
            parsed_answer = item["output_parsed_answer"]

            # Skip obviously corrupted data
            if len(parsed_answer) > 100:  # Arbitrary threshold for suspicious answers
                print(
                    f"\nSkipping suspiciously long parsed answer: {parsed_answer[:100]}..."
                )
                continue

            our_parsed = extract_answer(prediction_text)

            # Try to convert parsed answer to float, skip if it fails
            try:
                llama_parsed = float(parsed_answer.replace(",", ""))
                # llama_parsed = parse_number(parsed_answer)
            except ValueError:
                print(f"\nSkipping invalid parsed answer: {parsed_answer}")
                continue

            # Check if our parsing matches their parsing
            is_match = (
                abs(our_parsed - llama_parsed) < 1e-6
                if our_parsed is not None
                else False
            )

            if not is_match:
                comparison_results.append(
                    {
                        "text": prediction_text,
                        "our_parsed": our_parsed,
                        "llama_parsed": llama_parsed,
                        "is_match": is_match,
                        "llama_is_correct": item["is_correct"],
                    }
                )

            if is_match:
                parsing_matches += 1

        except Exception as e:
            print("\nError processing item:")
            print(f"Full prediction text: {item['output_prediction_text']}")
            print(f"Parsed answer from dataset: {item['output_parsed_answer']}")
            print(f"Error: {str(e)}")
            raise e

    # Print results
    print(f"\nParsing Comparison Results:")
    print(f"Total samples: {total}")
    print(f"Parsing matches: {parsing_matches} ({parsing_matches/total*100:.2f}%)")

    # Save mismatches for analysis
    if comparison_results:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        mismatch_file = f"results/parsing_mismatches-{timestamp}.json"
        os.makedirs(os.path.dirname(mismatch_file), exist_ok=True)

        with open(mismatch_file, "w") as f:
            json.dump(comparison_results, f, indent=2, default=str)
        print(f"\nMismatches saved to: {mismatch_file}")


def compare_evals(tokenizer):
    dataset = load_dataset("gsm8k", "main")["test"]
    llama_evals = load_dataset(
        "meta-llama/Llama-3.2-1B-Instruct-evals",
        name="Llama-3.2-1B-Instruct-evals__gsm8k__details",
        split="latest",
    )

    total = len(llama_evals)
    template_matches = 0
    llama_correct = 0
    comparison_results = []

    for i, item in tqdm(enumerate(llama_evals)):
        try:
            # question = dataset[i]["question"]
            question = item["input_question"]

            formatted_questions = [
                # base prompt
                # {
                #     "role": "system",
                #     "content": instruction_prompt,
                # },
                # few shot examples
                *format_few_shot_examples(llama3_2_gsm8k_few_shot_examples),
                {"role": "user", "content": format_question_prompt(question)},
            ]
            templated_questions = tokenizer.apply_chat_template(
                formatted_questions, tokenize=False, add_generation_prompt=True
            )

            # Print context when processing each item
            input_final_prompts = item["input_final_prompts"]

            prediction_text = item["output_prediction_text"][0]
            parsed_answer = item["output_parsed_answer"]
            our_parsed = extract_answer(prediction_text)

            # check matching templated questions -- doesn't seem to work
            # or could be some small difference?
            is_match = input_final_prompts == templated_questions

            if not is_match:
                comparison_results.append(
                    {
                        "our_templated": templated_questions,
                        "llama_template": input_final_prompts,
                        "is_template_match": is_match,
                        "llama_prediction": prediction_text,
                        "our_parsed": our_parsed,
                        "llama_parsed": parsed_answer,
                        "llama_is_correct": item["is_correct"],
                    }
                )
            if item["is_correct"]:
                llama_correct += 1

            if is_match:
                template_matches += 1

        except Exception as e:
            print("\nError processing item:")
            print(f"Full prediction text: {item['output_prediction_text']}")
            print(f"Parsed answer from dataset: {item['output_parsed_answer']}")
            print(f"Error: {str(e)}")
            raise e

    # Print results
    print(f"\nParsing Comparison Results:")
    print(f"Total samples: {total}")
    print(
        f"Llama correct predictions: {llama_correct} ({llama_correct/total*100:.2f}%)"
    )
    print(f"Parsing matches: {template_matches} ({template_matches/total*100:.2f}%)")

    # Save mismatches for analysis
    if comparison_results:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        mismatch_file = f"results/template_mismatches-{timestamp}.json"
        os.makedirs(os.path.dirname(mismatch_file), exist_ok=True)

        with open(mismatch_file, "w") as f:
            json.dump(comparison_results, f, indent=2, default=str)
        print(f"\nMismatches saved to: {mismatch_file}")
