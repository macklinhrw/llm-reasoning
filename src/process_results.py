import json


def process_results(path: str):
    with open(path, "r") as f:
        # load the json file
        results = json.load(f)

    # get the first 10 results
    # results = results[:10]

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
                            "question": question[i],
                            "response": response[i],
                            "target": target[i],
                            "predicted": predicted[i],
                        }
                    )
                )
                f.write("\n")


if __name__ == "__main__":
    process_results(
        "results/evaluation_results-Llama-3.1-8B-Instruct-2024-12-02-07-36.json"
    )
