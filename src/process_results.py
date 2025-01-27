import json
import os
from typing import Dict, List
from datasets import load_dataset


def load_gsm8k_solutions():
    """Load GSM8K dataset and create a mapping of questions to solutions."""
    dataset = load_dataset("gsm8k", "main")
    solutions = {}
    for item in dataset["test"]:
        solutions[item["question"]] = item["answer"]
    return solutions


def analyze_generations(results: List[Dict]) -> Dict:
    """Analyze the generations for each problem and calculate statistics."""
    # Load GSM8K solutions
    gsm8k_solutions = load_gsm8k_solutions()

    stats = {
        "total_problems": len(results),
        "total_generations": 0,
        "correct_generations": 0,
        "problems_with_correct_majority": 0,
    }

    problem_details = []

    for result in results:
        # Get all responses for this problem
        responses = result["all_responses"]
        correct_answer = result["correct"]
        question = result["question"]

        # Get solution from GSM8K
        solution = gsm8k_solutions.get(question)

        # Count generations for this problem
        num_generations = len(responses)
        stats["total_generations"] += num_generations

        # Count how many generations got the correct answer
        num_correct = sum(
            count
            for ans, count in result["answer_counts"].items()
            if abs(float(ans) - correct_answer) < 1e-6
        )
        stats["correct_generations"] += num_correct

        # Check if majority answer was correct
        majority_correct = abs(result["majority_answer"] - correct_answer) < 1e-6
        if majority_correct:
            stats["problems_with_correct_majority"] += 1

        # Store problem-level details
        problem_details.append(
            {
                "question": question,
                "correct_answer": correct_answer,
                "solution": solution,  # Add GSM8K solution
                "num_generations": num_generations,
                "num_correct": num_correct,
                "accuracy": num_correct / num_generations,
                "majority_answer": result["majority_answer"],
                "majority_correct": majority_correct,
                "answer_distribution": result["answer_counts"],
                "all_responses": result["all_responses"],  # Include all generations
                "prompt": result.get("prompt", None),  # Include prompt if available
            }
        )

    # Calculate overall statistics
    stats["generation_accuracy"] = (
        stats["correct_generations"] / stats["total_generations"]
    )
    stats["majority_accuracy"] = (
        stats["problems_with_correct_majority"] / stats["total_problems"]
    )

    return stats, problem_details


def save_incorrect_results(path: str):
    """Save only incorrect results to a JSONL file."""
    with open(path, "r") as f:
        data = json.load(f)
        results = data["results"]

    # get the input and output for each result
    question = [result["question"] for result in results]
    response = [result["response"] for result in results]
    is_correct = [result["is_correct"] for result in results]
    target = [result["correct"] for result in results]
    predicted = [result["predicted"] for result in results]

    # save only incorrect results to new file in data/processed
    with open("results/processed/incorrect_results.jsonl", "w+") as f:
        for i in range(len(question)):
            if not is_correct[i]:
                f.write(
                    json.dumps(
                        {
                            "index": i,
                            "question": question[i],
                            "response": response[i],
                            "target": target[i],
                            "predicted": predicted[i],
                        }
                    )
                )
                f.write("\n")


def process_results(path: str):
    """Process results file and save analysis."""
    # Load results
    with open(path, "r") as f:
        data = json.load(f)

    # Get model parameters and results
    params = data["parameters"]
    results = data["results"]

    # Analyze generations
    stats, problem_details = analyze_generations(results)

    # Prepare output
    output = {
        "model_name": params["model_name"],
        "generation_method": params["generation_method"],
        "timestamp": params["timestamp"],
        "statistics": stats,
        "problem_details": problem_details,  # Now includes all_responses for each problem
    }

    # Create analysis directory if it doesn't exist
    os.makedirs("results/analysis", exist_ok=True)

    # Save analysis results in analysis subfolder
    filename = os.path.basename(path)
    output_path = os.path.join(
        "results/analysis", filename.replace(".json", "-analysis.json")
    )
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary statistics
    print(f"\nAnalysis Results for {params['model_name']}:")
    print(f"Total problems: {stats['total_problems']}")
    print(f"Total generations: {stats['total_generations']}")
    print(f"Generation accuracy: {stats['generation_accuracy']:.2%}")
    print(f"Majority vote accuracy: {stats['majority_accuracy']:.2%}")


def filter_by_difficulty(
    path: str, threshold: float = None, top_n: int = None
) -> List[Dict]:
    """
    Filter problems by difficulty (1 - accuracy).

    Args:
        path: Path to results file
        threshold: Optional float between 0 and 1 to filter by difficulty percentage (e.g., 0.25 for top 25%)
        top_n: Optional int to filter by number of problems (e.g., 100 for top 100 hardest problems)

    Note: Provide either threshold or top_n, not both.

    Returns:
        List of difficult problems with their indices and difficulty scores
    """
    # Load GSM8K solutions
    gsm8k_solutions = load_gsm8k_solutions()

    if threshold is not None and top_n is not None:
        raise ValueError("Provide either threshold or top_n, not both")

    # Load and analyze results
    with open(path, "r") as f:
        data = json.load(f)

    results = data["results"]

    # Calculate difficulty for each problem
    difficult_problems = []
    for idx, result in enumerate(results):
        num_generations = len(result["all_responses"])
        num_correct = sum(
            count
            for ans, count in result["answer_counts"].items()
            if abs(float(ans) - result["correct"]) < 1e-6
        )
        accuracy = num_correct / num_generations
        difficulty = 1 - accuracy

        problem_info = {
            "index": idx,
            "question": result["question"],
            "correct_answer": result["correct"],
            "solution": gsm8k_solutions.get(result["question"]),  # Add GSM8K solution
            "difficulty": difficulty,
            "accuracy": accuracy,
            "num_generations": num_generations,
            "num_correct": num_correct,
            "answer_distribution": result["answer_counts"],
            "all_responses": result["all_responses"],  # Include all generations
            "prompt": result.get("prompt", None),  # Include prompt if available
        }
        difficult_problems.append(problem_info)

    # Sort by difficulty (descending)
    difficult_problems.sort(key=lambda x: x["difficulty"], reverse=True)

    # Filter based on provided criteria
    if threshold is not None:
        n = int(len(difficult_problems) * threshold)
        filtered_problems = difficult_problems[:n]
    elif top_n is not None:
        filtered_problems = difficult_problems[: min(top_n, len(difficult_problems))]
    else:
        filtered_problems = difficult_problems

    # Create difficulty directory
    os.makedirs("results/analysis/difficulty", exist_ok=True)

    # Save filtered problems
    filename = os.path.basename(path)
    base_name = filename.replace(".json", "")

    filter_desc = (
        f"top{top_n}"
        if top_n
        else f"top{int(threshold*100)}pct" if threshold else "all"
    )
    output_path = os.path.join(
        "results/analysis/difficulty", f"{base_name}-difficult-{filter_desc}.json"
    )

    output = {
        "model_name": data["parameters"]["model_name"],
        "generation_method": data["parameters"]["generation_method"],
        "filter_type": "top_n" if top_n else "threshold" if threshold else "none",
        "filter_value": top_n if top_n else threshold if threshold else None,
        "num_problems": len(filtered_problems),
        "problems": filtered_problems,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\nDifficulty Analysis:")
    print(f"Total problems analyzed: {len(difficult_problems)}")
    print(f"Problems selected: {len(filtered_problems)}")
    print(
        f"Difficulty range: {filtered_problems[0]['difficulty']:.2%} - {filtered_problems[-1]['difficulty']:.2%}"
    )
    print(f"Results saved to: {output_path}")

    return filtered_problems


if __name__ == "__main__":
    results_path = "results/evaluation_results-Qwen2.5-7B-Instruct-self_consistency_generate-2025-01-08-04-39.json"

    # Run analyses
    process_results(results_path)

    # Filter difficult problems (examples)
    # Get top 100 most difficult problems
    hard_problems = filter_by_difficulty(results_path, top_n=100)
    # Or get top 25% most difficult problems
    # hard_problems = filter_by_difficulty(results_path, threshold=0.25)
