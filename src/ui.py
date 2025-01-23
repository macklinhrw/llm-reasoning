import streamlit as st
from datasets import load_dataset
import json
import glob
import os


@st.cache_resource
def load_gsm8k_data():
    """Load GSM8K dataset from Hugging Face."""
    try:
        dataset = load_dataset("gsm8k", "main")
        return {"train": list(dataset["train"]), "test": list(dataset["test"])}
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None


def get_result_files(directory="results"):
    """Get all available result files from different subdirectories."""
    file_patterns = {
        "Incorrect Results": os.path.join(directory, "incorrect", "*.jsonl"),
        "Analysis Results": os.path.join(directory, "analysis", "*.json"),
        "Difficulty Analysis": os.path.join(
            directory, "analysis", "difficulty", "*.json"
        ),
    }

    files = {}
    for category, pattern in file_patterns.items():
        category_files = glob.glob(pattern)
        if category_files:
            # Sort files by modification time (newest first)
            category_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            files[category] = {os.path.basename(f): f for f in category_files}

    return files


@st.cache_resource
def load_results_file(file_path):
    """Load results from a single JSON/JSONL file."""
    results = []
    try:
        if file_path.endswith(".jsonl"):
            with open(file_path, "r") as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))
        else:
            with open(file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
                elif isinstance(data, dict):
                    results.append(data)
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")

    return results


def display_analysis_results(data):
    """Display analysis results in a structured way."""
    st.subheader("Model Information")
    st.write(f"Model: {data['model_name']}")
    st.write(f"Generation Method: {data['generation_method']}")

    # Handle both analysis and difficulty files
    if "statistics" in data:
        # Analysis file
        st.subheader("Statistics")
        stats = data["statistics"]
        cols = st.columns(2)

        with cols[0]:
            st.metric("Total Problems", stats["total_problems"])
            st.metric("Generation Accuracy", f"{stats['generation_accuracy']:.2%}")
        with cols[1]:
            st.metric("Total Generations", stats["total_generations"])
            st.metric("Majority Vote Accuracy", f"{stats['majority_accuracy']:.2%}")
    else:
        # Difficulty file
        st.subheader("Filter Information")
        st.write(f"Filter Type: {data['filter_type']}")
        if data["filter_value"] is not None:
            value = (
                f"Top {data['filter_value']}"
                if data["filter_type"] == "top_n"
                else f"Top {data['filter_value']*100:.0f}%"
            )
            st.write(f"Filter Value: {value}")
        st.metric("Number of Problems", data["num_problems"])

        # Calculate and display aggregate statistics for filtered problems
        if "problems" in data:
            total_generations = sum(p["num_generations"] for p in data["problems"])
            total_correct = sum(p["num_correct"] for p in data["problems"])
            accuracy = total_correct / total_generations if total_generations > 0 else 0

            cols = st.columns(2)
            with cols[0]:
                st.metric("Total Generations", total_generations)
                st.metric("Generation Accuracy", f"{accuracy:.2%}")
            with cols[1]:
                st.metric(
                    "Average Difficulty",
                    f"{sum(p['difficulty'] for p in data['problems'])/len(data['problems']):.2%}",
                )


def display_problem_details(problem, show_generations=False):
    """Display problem details in a structured way."""
    st.subheader("Problem")
    st.write(problem["question"])

    # Display metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Correct Answer", problem["correct_answer"])
        st.metric("Accuracy", f"{problem['accuracy']:.2%}")
    with col2:
        if "majority_answer" in problem:
            st.metric("Majority Answer", problem["majority_answer"])
        st.metric(
            "Difficulty", f"{problem.get('difficulty', 1-problem['accuracy']):.2%}"
        )
    with col3:
        st.metric(
            "Correct Generations",
            f"{problem['num_correct']}/{problem['num_generations']}",
        )

    # Display answer distribution if available
    if "answer_distribution" in problem:
        st.subheader("Answer Distribution")
        dist_data = problem["answer_distribution"]
        st.bar_chart(dist_data)

    # Display generations if available and requested
    if show_generations and "all_responses" in problem:
        st.subheader("All Generations")
        for i, response in enumerate(problem["all_responses"], 1):
            with st.expander(f"Generation {i}"):
                st.code(response, wrap_lines=True)

    # Display prompt if available
    if "prompt" in problem and problem["prompt"]:
        with st.expander("Prompt"):
            st.code(problem["prompt"], wrap_lines=True)


def main():
    st.set_page_config(page_title="Dataset Viewer", page_icon="🔢", layout="wide")
    st.title("Dataset Viewer")

    # Initialize session state
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
        st.session_state.input_idx = 0

    # Data source selection
    data_source = st.sidebar.selectbox(
        "Choose data source",
        ["GSM8K Dataset", "Model Results", "Analysis Results", "Difficulty Analysis"],
    )

    if data_source == "GSM8K Dataset":
        with st.spinner("Loading dataset from Hugging Face..."):
            dataset = load_gsm8k_data()

        if dataset is None:
            st.warning("Failed to load dataset.")
            return

        split = st.sidebar.selectbox("Choose dataset split", ["train", "test"])
        examples = dataset[split]

    elif data_source in ["Analysis Results", "Difficulty Analysis"]:
        # Get available result files
        all_files = get_result_files()
        category = (
            "Analysis Results"
            if data_source == "Analysis Results"
            else "Difficulty Analysis"
        )

        if category not in all_files:
            st.warning(f"No {category.lower()} files found.")
            return

        file_options = all_files[category]
        selected_file = st.sidebar.selectbox(
            "Choose file",
            options=list(file_options.keys()),
            format_func=lambda x: x.replace(".json", ""),
        )

        with st.spinner("Loading analysis..."):
            with open(file_options[selected_file], "r") as f:
                data = json.load(f)

            # Display analysis overview
            display_analysis_results(data)

            # Problem navigation
            problems = (
                data["problem_details"]
                if "problem_details" in data
                else data["problems"]
            )
            examples = problems  # Use common variable name for navigation

    else:  # Original Model Results
        # Get available result files
        file_options = get_result_files()
        if not file_options:
            st.warning("No result files found.")
            return

        # Let user select which file to load
        selected_file = st.sidebar.selectbox(
            "Choose results file",
            options=list(file_options.keys()),
            format_func=lambda x: x.replace("results/", "")
            .replace(".json", "")
            .replace(".jsonl", ""),
        )

        with st.spinner("Loading results..."):
            examples = load_results_file(file_options[selected_file])
            if not examples:
                st.warning("No results found in selected file.")
                return

    total_examples = len(examples)
    st.sidebar.markdown(f"Total examples: {total_examples}")

    # Navigation controls
    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

    with col1:
        if st.button("⬅️ Prev"):
            st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
            st.session_state.input_idx = st.session_state.current_idx

    with col2:
        if st.button("Next ➡️"):
            st.session_state.current_idx = min(
                total_examples - 1, st.session_state.current_idx + 1
            )
            st.session_state.input_idx = st.session_state.current_idx

    with col4:
        st.number_input(
            "Go to",
            min_value=0,
            max_value=total_examples - 1,
            key="input_idx",
            label_visibility="collapsed",
        )
        if st.session_state.input_idx != st.session_state.current_idx:
            st.session_state.current_idx = st.session_state.input_idx

    # Display progress
    st.progress(st.session_state.current_idx / (total_examples - 1))
    st.markdown(f"**Example {st.session_state.current_idx + 1} of {total_examples}**")

    # Display current example
    if 0 <= st.session_state.current_idx < total_examples:
        example = examples[st.session_state.current_idx]

        if data_source in ["Analysis Results", "Difficulty Analysis"]:
            show_generations = st.checkbox("Show all generations")
            display_problem_details(example, show_generations)
        elif data_source == "GSM8K Dataset":
            st.subheader("Problem")
            st.code(example["question"], language=None)
            st.subheader("Solution")
            st.code(example["answer"], language=None)
        else:  # Model Results
            st.subheader("Problem")
            st.code(example.get("question", "N/A"), language=None, wrap_lines=True)

            if "response" in example:
                st.subheader("Model Response")
                st.code(example["response"], language=None, wrap_lines=True)

            if "target" in example and "predicted" in example:
                st.subheader("Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Target", example["target"])
                with col2:
                    st.metric("Predicted", example["predicted"])

        # Display all metadata
        with st.expander("Raw Data"):
            st.json(example)


if __name__ == "__main__":
    main()
