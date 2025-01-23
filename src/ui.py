import streamlit as st
from datasets import load_dataset
import json
import glob
import os
import re
from datetime import datetime
from utils import extract_answer


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
        "Model Results": [
            os.path.join(directory, "*.json"),
            os.path.join(directory, "*.jsonl")
        ],
        "Incorrect Results": os.path.join(directory, "incorrect", "*.jsonl"),
        "Analysis Results": os.path.join(directory, "analysis", "*.json"),
        "Difficulty Analysis": os.path.join(directory, "analysis", "difficulty", "*.json"),
    }

    files = {}
    for category, patterns in file_patterns.items():
        if isinstance(patterns, str):
            patterns = [patterns]
        
        category_files = []
        for pattern in patterns:
            category_files.extend(glob.glob(pattern))

        if category_files:
            # Sort files by modification time (newest first)
            category_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            files[category] = {os.path.basename(f): f for f in category_files}

    return files


@st.cache_resource
def load_results_file(file_path):
    """Load results from a single JSON/JSONL file with flexible format handling."""
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
                
                # Handle different JSON structures
                if isinstance(data, list):
                    results = data
                elif "results" in data:  # Format with metadata + results list
                    results = data["results"]
                elif "problem_details" in data:  # Analysis format
                    results = data["problem_details"]
                else:  # Single result format
                    results.append(data)

    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")

    return results


def parse_result_filename(filename):
    """Extract metadata from result filenames."""
    metadata = {
        "full_name": filename,
        "display_name": filename.replace("evaluation_results-", "").replace(".json", ""),
        "model": "Unknown Model"  # Add default value
    }
    
    # Improved regex pattern with fallbacks
    try:
        # First try to extract model name from filename
        model_match = re.search(r"(?:evaluation_results-)?([A-Za-z0-9_.-]+?)(?:-\d{4}|$)", filename)
        if model_match:
            model_raw = model_match.group(1)
            # Clean up model name
            model_parts = model_raw.replace("_", " ").split("-")
            # Format parts like "3.2" and "1B" nicely
            formatted_parts = []
            for part in model_parts:
                if part.lower().endswith("b"):
                    formatted_parts.append(part.upper())
                else:
                    formatted_parts.append(part.capitalize())
            metadata["model"] = " ".join(formatted_parts)
            
        # Special case for Qwen models
        if "qwen" in filename.lower():
            metadata["model"] = filename.split("-")[0].upper()
            
    except Exception as e:
        st.error(f"Error parsing filename {filename}: {str(e)}")

    # Extract date from filename
    try:
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
        if date_match:
            metadata["date"] = datetime.strptime(date_match.group(1), "%Y-%m-%d").strftime("%b %d, %Y")
        else:
            metadata["date"] = "Unknown Date"
    except Exception as e:
        metadata["date"] = "Invalid Date"
        st.error(f"Error parsing date in {filename}: {str(e)}")

    return metadata


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
    st.code(problem["question"], language='text', wrap_lines=True)
    
    if "solution" in problem:
        st.subheader("Solution")
        st.code(problem["solution"], language='text', wrap_lines=True)

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
        
        # Add filtering controls
        col1, col2 = st.columns(2)
        with col1:
            filter_type = st.selectbox("Filter generations", 
                                     ["All", "Correct", "Incorrect"],
                                     key=f"filter_{problem['question']}")
        with col2:
            show_solution = st.checkbox("Show solution alongside", 
                                      key=f"solution_toggle_{problem['question']}")

        # Precompute correctness for performance
        correct_answer = problem["correct_answer"]
        responses = []
        for response in problem["all_responses"]:
            extracted_answer = extract_answer(response)
            is_correct = abs(extracted_answer - correct_answer) < 1e-6 if extracted_answer else False
            responses.append((response, is_correct))

        # Apply filter
        if filter_type == "Correct":
            filtered = [r for r in responses if r[1]]
        elif filter_type == "Incorrect":
            filtered = [r for r in responses if not r[1]]
        else:
            filtered = responses

        # Display generations with solution comparison
        for idx, (response, is_correct) in enumerate(filtered, 1):
            status = "✅" if is_correct else "❌"
            with st.expander(f"Generation {idx} {status}", expanded=False):
                if show_solution and "solution" in problem:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Model Response**")
                        st.code(response, language='markdown', wrap_lines=True)
                    with col2:
                        st.markdown("**Reference Solution**")
                        st.code(problem["solution"], language='text', wrap_lines=True)
                else:
                    st.code(response, language='markdown', wrap_lines=True)

        # Show quick stats
        total_correct = sum(1 for r in responses if r[1])
        st.caption(f"Showing {len(filtered)}/{len(responses)} generations "
                 f"({total_correct} correct, {len(responses)-total_correct} incorrect)")

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

    else:  # Model Results
        # Get available result files
        all_files = get_result_files()
        category = "Model Results"
        
        if category not in all_files or not all_files[category]:
            st.warning(f"No {category} files found.")
            return

        # Parse and organize files
        file_data = [parse_result_filename(f) for f in all_files[category].keys()]
        
        # Create sidebar filters
        st.sidebar.subheader("Filter Results")
        
        with st.sidebar:
            with st.expander("🔍 Search Filters", expanded=True):
                search_term = st.text_input("Search files", "", 
                                          placeholder="Model name or date...",
                                          help="Search by model family, date, or method")
                
                col1, col2 = st.columns(2)
                with col1:
                    unique_models = sorted(
                        {fd.get("model", "Unknown Model") for fd in file_data},
                        key=lambda x: (
                            not x.startswith("Llama"),  # Llama first
                            not x.startswith("Qwen"),   # Then Qwen
                            x.split()[-1]               # Sort by model size
                        )
                    )
                    selected_model = st.selectbox(
                        "Model Family", ["All Models"] + unique_models, index=0
                    )
                with col2:
                    unique_dates = sorted({fd.get("date") for fd in file_data if fd.get("date")}, 
                                        reverse=True)
                    selected_date = st.selectbox(
                        "Date", ["All Dates"] + unique_dates, index=0
                    )

            st.markdown("---")
            st.markdown("**Selected File**")
            filtered_files = [
                f for f in file_data
                if (selected_model == "All Models" or f["model"] == selected_model) and
                (selected_date == "All Dates" or f.get("date") == selected_date) and
                search_term in f["full_name"].lower()
            ]
            selected_file = st.selectbox(
                "Choose results file",
                options=[f["full_name"] for f in filtered_files],
                format_func=lambda x: next(f["display_name"] for f in filtered_files if f["full_name"] == x),
                help="Files sorted by modification date (newest first)",
                label_visibility="collapsed"
            )
        
        if not selected_file:
            st.warning("No files match filters")
            return

        # Load results
        with st.spinner("Loading results..."):
            loaded_data = load_results_file(all_files[category][selected_file])
            examples = loaded_data  # For files that are just lists
            params = {}

            # If the data has a parameters field (structured format)
            if isinstance(loaded_data, dict) and "parameters" in loaded_data:
                params = loaded_data["parameters"]
                examples = loaded_data.get("results", [])

            if not examples:
                st.warning("No results found in selected file.")
                return

        # Summary header
        st.subheader("Results Summary")

        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Problems", len(examples))
        with cols[1]:
            if "final_accuracy" in params:
                st.metric("Accuracy", f"{params['final_accuracy']:.2f}%")
        with cols[2]:
            temp = params.get("generate_kwargs", {}).get("temperature", "N/A")
            st.metric("Temperature", temp)
        with cols[3]:
            method = params.get("generation_method", "N/A").replace("_", " ").title()
            st.metric("Method", method)
        
        st.markdown("---")  # Horizontal line

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
        elif data_source == "Model Results":
            tab1, tab2, tab3 = st.tabs(["Problem & Response", "Analysis", "Raw Data"])
            
            with tab1:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Problem Statement")
                    st.markdown(f"```\n{example.get('question', example.get('input_question', 'N/A'))}\n```")
                    
                    if "all_responses" in example:
                        with st.expander(f"View All Responses ({len(example['all_responses'])})", expanded=False):
                            for i, resp in enumerate(example["all_responses"], 1):
                                st.markdown(f"**Response {i}**")
                                st.code(resp, language="markdown")
                
                with col2:
                    st.subheader("Answers")
                    answer_col1, answer_col2 = st.columns(2)
                    
                    with answer_col1:
                        correct_answer = example.get("correct") or example.get("correct_answer") or example.get("target")
                        st.markdown(f"### Correct Answer")
                        if correct_answer is not None:
                            st.markdown(f"<h2 style='color: #2ecc71;'>{correct_answer}</h2>", unsafe_allow_html=True)
                        else:
                            st.warning("No correct answer found in data")

                    with answer_col2:
                        pred_answer = example.get("predicted_answer") or example.get("predicted") or "N/A"
                        is_correct = example.get("is_correct", False)
                        color = "#2ecc71" if is_correct else "#e74c3c"
                        st.markdown(f"### Model Prediction")
                        st.markdown(f"<h2 style='color: {color};'>{pred_answer}</h2>", unsafe_allow_html=True)
                    
                    if "confidence" in example:
                        st.metric("Confidence Score", f"{example['confidence']:.2%}",
                                help="Model's self-reported confidence in the answer")
            
            with tab2:
                if "analysis" in example:
                    st.subheader("Detailed Analysis")
                    st.write(example["analysis"])
                else:
                    st.write("No detailed analysis available for this example")
                
                if "error_types" in example:
                    st.subheader("Error Categories")
                    for error in example["error_types"]:
                        st.markdown(f"- {error}")
            
            with tab3:
                st.json(example)


if __name__ == "__main__":
    main()
