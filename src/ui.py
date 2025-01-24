import streamlit as st
from datasets import load_dataset
import json
import glob
import os
import re
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from utils import extract_answer

# Difficulty types configuration
DIFFICULTY_CONFIG_FILE = "config/difficulty_types.json"

def load_difficulty_types():
    """Load difficulty types from configuration file."""
    default_types = [
        "Arithmetic Complexity",
        "Multi-step Reasoning",
        "Ambiguous Wording",
        "Real-world Knowledge",
        "Unit Conversion",
        "Symbolic Reasoning",
        "Counterintuitive",
        "Other"
    ]
    
    try:
        os.makedirs(os.path.dirname(DIFFICULTY_CONFIG_FILE), exist_ok=True)
        if os.path.exists(DIFFICULTY_CONFIG_FILE):
            with open(DIFFICULTY_CONFIG_FILE, "r") as f:
                return json.load(f)
        # Save default types if file doesn't exist
        with open(DIFFICULTY_CONFIG_FILE, "w") as f:
            json.dump(default_types, f)
        return default_types
    except Exception as e:
        st.error(f"Error loading difficulty types: {str(e)}")
        return default_types

def save_difficulty_types(types):
    """Save updated difficulty types to configuration file."""
    try:
        with open(DIFFICULTY_CONFIG_FILE, "w") as f:
            json.dump(types, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving difficulty types: {str(e)}")
        return False

def get_difficulty_types():
    return load_difficulty_types()


@st.cache_resource
def load_gsm8k_data():
    """Load GSM8K dataset from Hugging Face."""
    try:
        dataset = load_dataset("gsm8k", "main")
        return {"train": list(dataset["train"]), "test": list(dataset["test"])}
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None


def get_annotation_path(data_file: str) -> str:
    """Get path for annotation file based on data filename."""
    base_name = os.path.basename(data_file).replace(".json", "")
    return os.path.join("annotations", f"{base_name}-annotations.json")

def load_annotations(data_file: str) -> Dict[str, dict]:
    """Load existing annotations for a data file."""
    annotation_path = get_annotation_path(data_file)
    if os.path.exists(annotation_path):
        with open(annotation_path, "r") as f:
            return json.load(f)
    return {}

def save_annotations(data_file: str, annotations: Dict[str, dict]):
    """Save annotations to file."""
    annotation_path = get_annotation_path(data_file)
    os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
    with open(annotation_path, "w") as f:
        json.dump(annotations, f, indent=2)

def get_result_files(directory="results"):
    """Get all available result files from different subdirectories."""
    file_patterns = {
        "Model Results": [
            os.path.join(directory, "*.json"),
            os.path.join(directory, "*.jsonl"),
        ],
        "Incorrect Results": os.path.join(directory, "incorrect", "*.jsonl"),
        "Analysis Results": os.path.join(directory, "analysis", "*.json"),
        "Difficulty Analysis": os.path.join(
            directory, "analysis", "difficulty", "*.json"
        ),
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
        "display_name": filename.replace("evaluation_results-", "").replace(
            ".json", ""
        ),
        "model": "Unknown Model",  # Add default value
    }

    # Improved regex pattern with fallbacks
    try:
        # First try to extract model name from filename
        model_match = re.search(
            r"(?:evaluation_results-)?([A-Za-z0-9_.-]+?)(?:-\d{4}|$)", filename
        )
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
            metadata["date"] = datetime.strptime(
                date_match.group(1), "%Y-%m-%d"
            ).strftime("%b %d, %Y")
        else:
            metadata["date"] = "Unknown Date"
    except Exception as e:
        metadata["date"] = "Invalid Date"
        st.error(f"Error parsing date in {filename}: {str(e)}")

    return metadata


def display_analysis_results(data):
    """Display analysis results in a structured way with improved UI."""
    # Create tabs for different sections
    tab_overview, tab_details = st.tabs(["📊 Overview", "📝 Problem Details"])

    with tab_overview:
        st.subheader("Model Overview")
        cols = st.columns([1, 2])
        with cols[0]:
            st.metric("Model Name", data.get("model_name", "N/A"))
            st.metric("Generation Method", data.get("generation_method", "N/A").title())
        with cols[1]:
            st.metric("Evaluation Date", data.get("timestamp", "N/A"))
            st.metric(
                "Total Problems",
                data.get("statistics", {}).get("total_problems", "N/A"),
            )

        st.markdown("---")
        st.subheader("Key Metrics")

        # Create metric columns with icons
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "🏅 Generation Accuracy",
                (
                    f"{data['statistics']['generation_accuracy']:.2%}"
                    if "statistics" in data
                    else "N/A"
                ),
                help="Accuracy of individual generations",
            )
        with col2:
            st.metric(
                "👥 Majority Accuracy",
                (
                    f"{data['statistics']['majority_accuracy']:.2%}"
                    if "statistics" in data
                    else "N/A"
                ),
                help="Accuracy when using majority voting",
            )
        with col3:
            st.metric(
                "📚 Total Generations",
                (
                    data["statistics"].get("total_generations", "N/A")
                    if "statistics" in data
                    else "N/A"
                ),
            )
        with col4:
            st.metric(
                "🎚️ Temperature",
                (
                    data["parameters"].get("temperature", "N/A")
                    if "parameters" in data
                    else "N/A"
                ),
            )

        # Add visualization section
        if "statistics" in data:
            st.markdown("---")
            st.subheader("Accuracy Distribution")

            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                st.markdown("**By Problem Difficulty**")
                # Example visualization - would need actual difficulty data
                st.bar_chart({"Low": 0.85, "Medium": 0.72, "High": 0.55})

            with col_viz2:
                st.markdown("**Temporal Accuracy Trend**")
                # Example time-based trend
                st.line_chart({"Run 1": 0.68, "Run 2": 0.72, "Run 3": 0.75})

    with tab_details:
        st.subheader("Problem-Level Analysis")
        st.caption("Browse individual problems with detailed performance metrics")

        # Add filtering controls
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            min_accuracy = st.slider("Minimum accuracy", 0.0, 1.0, 0.0, 0.01)
        with col_filter2:
            show_incorrect = st.checkbox("Show only incorrect problems", value=False)

        # Problem cards
        for idx, problem in enumerate(data.get("problem_details", [])):
            # Apply filters
            if problem["accuracy"] < min_accuracy:
                continue
            if show_incorrect and problem["accuracy"] == 1.0:
                continue

            # Create expandable problem card
            with st.expander(
                f"Problem {idx+1} | Accuracy: {problem['accuracy']:.2%}", expanded=False
            ):
                display_problem_card(problem)


def display_problem_card(problem):
    """Display a problem in a condensed card format with visual indicators."""
    annotation_status = "📌" if problem.get("annotations") else ""
    with st.expander(
        f"{annotation_status} Problem | Accuracy: {problem['accuracy']:.2%}", 
        expanded=False
    ):
        cols = st.columns([3, 1, 1, 1])
    
    # Add annotation display
    if "annotations" in problem:
        st.markdown("**Annotations**")
        cols = st.columns(2)
        with cols[0]:
            if problem["annotations"].get("difficulty_types"):
                st.write("Types: " + ", ".join(problem["annotations"]["difficulty_types"]))
        with cols[1]:
            if problem["annotations"].get("notes"):
                st.write(f"Notes: {problem['annotations']['notes']}")

    # Calculate derived difficulty if not present
    derived_difficulty = 1 - problem.get("accuracy", 0)

    with cols[0]:
        st.markdown(f"**Question**  \n{problem['question'][:200]}...")

    with cols[1]:
        st.metric("Correct", problem["correct_answer"])
    with cols[2]:
        pred = problem.get("predicted_answer", "N/A")
        st.metric("Predicted", pred, delta="✅" if problem.get("is_correct") else "❌")
    with cols[3]:
        # Show accuracy instead of difficulty when they're directly related
        st.metric("Accuracy", f"{problem['accuracy']:.2%}")

    # Visual indicator now shows accuracy
    accuracy = problem["accuracy"]
    st.markdown(
        f"""
    <style>
        .accuracy-bar {{
            height: 8px;
            background: linear-gradient(90deg, #2ecc71 {accuracy*100}%, #e74c3c {accuracy*100}%);
            border-radius: 4px;
            margin: 8px 0;
        }}
    </style>
    <div class="accuracy-bar"></div>
    """,
        unsafe_allow_html=True,
    )

    # Update detailed view
    with st.expander("Detailed Analysis", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if "all_responses" in problem:
                st.markdown("**Answer Distribution**")
                if problem.get("answer_distribution"):
                    st.bar_chart(problem["answer_distribution"])
        with col2:
            st.markdown("**Performance Metrics**")
            st.write(
                f"- Correct Generations: {problem['num_correct']}/{problem['num_generations']}"
            )
            if "difficulty" in problem:
                st.write(f"- Difficulty: {problem['difficulty']:.2%}")
            else:
                st.write(f"- Difficulty (1 - Accuracy): {derived_difficulty:.2%}")

    st.markdown("---")


def display_problem_details(problem, show_generations=False):
    """Display problem details in a structured way."""
    st.subheader("Problem")
    st.code(problem["question"], language="text", wrap_lines=True)

    if "solution" in problem:
        st.subheader("Solution")
        st.code(problem["solution"], language="text", wrap_lines=True)

    # Calculate derived difficulty if needed
    derived_difficulty = 1 - problem["accuracy"]

    # Update metrics display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Correct Answer", problem["correct_answer"])
        st.metric("Accuracy", f"{problem['accuracy']:.2%}")
    with col2:
        if "majority_answer" in problem:
            st.metric("Majority Answer", problem["majority_answer"])
        if "difficulty" in problem:
            st.metric("Difficulty", f"{problem['difficulty']:.2%}")
        else:
            st.metric(
                "Derived Difficulty",
                f"{derived_difficulty:.2%}",
                help="Calculated as 1 - Accuracy",
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

        # Precompute correctness for performance
        correct_answer = problem["correct_answer"]
        responses = []
        for response in problem["all_responses"]:
            extracted_answer = extract_answer(response)
            is_correct = (
                abs(extracted_answer - correct_answer) < 1e-6
                if extracted_answer
                else False
            )
            responses.append((response, is_correct))

        # Show stats at top
        total_correct = sum(1 for r in responses if r[1])
        st.caption(
            f"Showing {len(responses)} generations "
            f"({total_correct} correct, {len(responses)-total_correct} incorrect)"
        )

        # Add filtering controls
        col1, col2 = st.columns(2)
        with col1:
            filter_type = st.selectbox(
                "Filter generations",
                ["All", "Correct", "Incorrect"],
                key=f"filter_{problem['question']}",
            )
        with col2:
            show_solution = st.checkbox(
                "Show solution alongside", key=f"solution_toggle_{problem['question']}"
            )

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
                        st.code(response, language="markdown", wrap_lines=True)
                    with col2:
                        st.markdown("**Reference Solution**")
                        st.code(problem["solution"], language="text", wrap_lines=True)
                else:
                    st.code(response, language="markdown", wrap_lines=True)

    # Display prompt if available
    if "prompt" in problem and problem["prompt"]:
        with st.expander("Prompt"):
            st.code(problem["prompt"], wrap_lines=True)


def main():
    st.set_page_config(page_title="Dataset Viewer", page_icon="🔢", layout="wide")
    st.title("Dataset Viewer")

    # Initialize session state
    if "pending_deletion" not in st.session_state:
        st.session_state.pending_deletion = None
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

            # New layout
            st.subheader("🧩 Difficulty Analysis")

            # Summary banner
            cols = st.columns(4)
            with cols[0]:
                st.metric("Filter Type", data["filter_type"].title())
            with cols[1]:
                value = (
                    f"Top {data['filter_value']}"
                    if data["filter_type"] == "top_n"
                    else f"Top {data['filter_value']*100:.0f}%"
                )
                st.metric("Filter Value", value)
            with cols[2]:
                st.metric("Problems Found", data["num_problems"])
            with cols[3]:
                avg_difficulty = sum(p["difficulty"] for p in data["problems"]) / len(
                    data["problems"]
                )
                st.metric("Avg Difficulty", f"{avg_difficulty:.2%}")

            # Visualization row
            st.markdown("---")
            st.markdown("**Difficulty Distribution**")
            difficulties = [
                1 - p["accuracy"] for p in data["problems"]
            ]  # Calculate difficulty
            bin_edges = np.linspace(0, 1, 21)
            hist, bin_edges = np.histogram(difficulties, bins=bin_edges)
            bin_labels = [
                f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}"
                for i in range(len(bin_edges) - 1)
            ]

            chart_df = pd.DataFrame(
                {"Difficulty Range": bin_labels, "Number of Problems": hist}
            ).set_index("Difficulty Range")

            st.bar_chart(chart_df, use_container_width=True)

            # Sorting controls (removed accuracy option)
            col_sort1, col_sort2 = st.columns(2)
            with col_sort1:
                sort_by = st.selectbox(
                    "Sort by", 
                    ["Difficulty"],
                    key="sort_by_1"  # Unique key
                )
            with col_sort2:
                sort_order = st.selectbox(
                    "Order", 
                    ["Descending", "Ascending"],
                    key="sort_order_1"  # Unique key
                )

            # Sort problems and persist in session state
            if "sorted_examples" not in st.session_state or \
                st.session_state.get("current_sort") != (sort_by, sort_order):
                
                reverse = sort_order == "Descending"
                key = "difficulty" if sort_by == "Difficulty" else "accuracy"
                
                st.session_state.sorted_examples = sorted(
                    data["problems"],
                    key=lambda x: x[key],
                    reverse=reverse
                )
                st.session_state.current_sort = (sort_by, sort_order)
                st.session_state.current_idx = 0  # Reset index when sorting changes

            examples = st.session_state.sorted_examples

            # Create tabs AFTER initializing examples
            tab_details, tab_annotate, tab_browser = st.tabs([
                "🔍 Problem Details", 
                "🏷️ Annotation", 
                "📚 Annotations Browser"
            ])

            with tab_annotate:
                st.subheader("Problem Annotation")
                
                # Initialize annotations in session state
                if "annotations" not in st.session_state:
                    st.session_state.annotations = load_annotations(file_options[selected_file])
                
                # Get current problem from persisted examples
                if "sorted_examples" not in st.session_state:
                    st.session_state.sorted_examples = examples
                
                # Get fresh reference to the current problem
                current_idx = st.session_state.current_idx
                examples = st.session_state.sorted_examples  # Get fresh sorted list
                current_problem = examples[current_idx]
                problem_id = current_problem["question"]

                # Add bounds checking
                if current_idx >= len(examples):
                    st.session_state.current_idx = len(examples) - 1
                    st.rerun()

                # Clear widget states when problem changes
                if "last_problem_id" not in st.session_state:
                    st.session_state.last_problem_id = problem_id
                
                if st.session_state.last_problem_id != problem_id:
                    # Clear previous widget states using hashed keys
                    prev_id = st.session_state.last_problem_id
                    keys_to_remove = [
                        f"types_{hash(prev_id)}",
                        f"notes_{hash(prev_id)}"
                    ]
                    for key in keys_to_remove:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.session_state.last_problem_id = problem_id
                
                # Get existing annotation or create new
                annotation = st.session_state.annotations.get(problem_id, {
                    "difficulty_types": [],
                    "notes": ""
                })

                # Difficulty type selection
                difficulty_types = load_difficulty_types()

                # Add type management controls
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected_types = st.multiselect(
                        "Difficulty Types",
                        difficulty_types,
                        default=annotation["difficulty_types"],
                        key=f"types_{hash(problem_id)}"  # Use hash for shorter unique key
                    )

                with col2:
                    if st.button("✏️ Edit Types", help="Modify available difficulty types"):
                        st.session_state.edit_types = True

                if st.session_state.get('edit_types'):
                    with st.expander("📝 Edit Difficulty Types", expanded=True):
                        new_types = st.text_area(
                            "Edit difficulty types (comma-separated)",
                            value=", ".join(difficulty_types),
                            height=150
                        )
                        
                        if st.button("💾 Save Types"):
                            updated_types = [t.strip() for t in new_types.split(",") if t.strip()]
                            if save_difficulty_types(updated_types):
                                st.session_state.difficulty_types = updated_types
                                st.success("Types updated!")
                                st.session_state.edit_types = False
                                st.rerun()
                        
                        if st.button("❌ Cancel"):
                            st.session_state.edit_types = False
                            st.rerun()  # Force immediate UI refresh
                
                # Notes field
                notes = st.text_area(
                    "Additional Notes",
                    value=annotation["notes"],
                    height=150,
                    key=f"notes_{hash(problem_id)}"  # Use hash here too
                )
                
                # Save button
                if st.button("💾 Save Annotation"):
                    st.session_state.annotations[problem_id] = {
                        "difficulty_types": selected_types,
                        "notes": notes,
                        "source_file": os.path.basename(file_options[selected_file])
                    }
                    save_annotations(file_options[selected_file], st.session_state.annotations)
                    st.success("Annotation saved!")
                
                # Add delete button for existing annotations
                if problem_id in st.session_state.annotations:
                    st.markdown("---")
                    if st.button("🗑️ Delete Annotation", type="primary", key=f"delete_{hash(problem_id)}"):
                        st.session_state.pending_deletion = problem_id

                # Handle deletion confirmation
                if "pending_deletion" in st.session_state and st.session_state.pending_deletion == problem_id:
                    st.warning("Confirm deletion?")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Confirm Delete"):
                            del st.session_state.annotations[st.session_state.pending_deletion]
                            save_annotations(file_options[selected_file], st.session_state.annotations)
                            del st.session_state.pending_deletion
                            st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
                            st.rerun()
                    with col2:
                        if st.button("❌ Cancel"):
                            del st.session_state.pending_deletion
                            st.rerun()
                

            with tab_browser:
                st.subheader("Annotations Browser")
                annotations = st.session_state.annotations

                if not annotations:
                    st.write("No annotations saved yet")
                else:
                    # Add filtering controls
                    st.markdown("---")
                    col_filter1, col_filter2 = st.columns(2)
                    with col_filter1:
                        search_query = st.text_input("Search annotations", "")
                    with col_filter2:
                        selected_type_filter = st.multiselect(
                            "Filter by type", 
                            difficulty_types,
                            default=[]
                        )
                    
                    # Group annotations by type
                    filtered_annotations = []
                    for q, ann in annotations.items():
                        matches_search = search_query.lower() in q.lower() or search_query.lower() in ann["notes"].lower()
                        matches_type = not selected_type_filter or any(t in selected_type_filter for t in ann["difficulty_types"])
                        if matches_search and matches_type:
                            filtered_annotations.append((q, ann))
                    
                    # Add pagination
                    items_per_page = 10
                    total_pages = (len(filtered_annotations) + items_per_page - 1) // items_per_page

                    # Handle empty results case
                    if total_pages > 0:
                        current_page = st.number_input(
                            "Page", 
                            min_value=1, 
                            max_value=total_pages, 
                            value=1
                        ) - 1
                    else:
                        current_page = 0
                    
                    start_idx = current_page * items_per_page
                    end_idx = min(start_idx + items_per_page, len(filtered_annotations))
                    
                    # Display annotation cards in columns
                    for idx, (q, ann) in enumerate(filtered_annotations[start_idx:end_idx]):
                        with st.expander(f"📌 {q[:50]}...", expanded=False):
                            cols = st.columns([3, 1])
                            with cols[0]:
                                st.caption("Problem Text")
                                st.markdown(f"```\n{q[:200]}...\n```")
                                
                                st.caption("Notes")
                                if ann["notes"]:
                                    st.markdown(f"> *{ann['notes'][:100]}...*")
                                else:
                                    st.markdown("*No notes*")
                            
                            with cols[1]:
                                st.caption("Difficulty Types")
                                if ann["difficulty_types"]:
                                    for t in ann["difficulty_types"]:
                                        st.markdown(f"- `{t}`")
                                else:
                                    st.markdown("*No types specified*")
                                
                                # Add delete button
                                unique_key = f"{idx}_{hash(q)}"  # Combine index and hash for uniqueness
                                if st.button("🗑️ Delete", key=f"delete_{unique_key}"):
                                    st.session_state.pending_deletion = q

                                # Handle deletion confirmation
                                if "pending_deletion" in st.session_state and st.session_state.pending_deletion == q:
                                    st.warning(f"Confirm deletion for: {q[:50]}...?")
                        
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button("✅ Confirm Delete", key=f"confirm_{unique_key}"):
                                            del st.session_state.annotations[q]
                                            save_annotations(file_options[selected_file], st.session_state.annotations)
                                            del st.session_state.pending_deletion
                                            st.rerun()
                                    with col2:
                                        if st.button("❌ Cancel", key=f"cancel_{unique_key}"):
                                            del st.session_state.pending_deletion
                                            st.rerun()
                    
                                if st.button("Go to Annotation", key=f"goto_{unique_key}"):
                                    problem_ids = [p["question"] for p in examples]
                                    if q in problem_ids:
                                        st.session_state.current_idx = problem_ids.index(q)
                                        st.rerun()
                                    else:
                                        st.warning("This annotation isn't in the current view")
                    
                    # Pagination controls
                    if total_pages > 1:
                        st.markdown(f"**Page {current_page + 1} of {total_pages}**")

            # Problem list
            st.markdown("---")
            st.subheader(f"Filtered Problems ({data['num_problems']})")

            # Sorting controls (removed accuracy option)
            col_sort1, col_sort2 = st.columns(2)
            with col_sort1:
                sort_by = st.selectbox(
                    "Sort by", 
                    ["Difficulty"],
                    key="sort_by_2"  # Unique key
                )
            with col_sort2:
                sort_order = st.selectbox(
                    "Order", 
                    ["Descending", "Ascending"],
                    key="sort_order_2"  # Unique key
                )

            # Add annotations to problems
            annotations = load_annotations(file_options[selected_file])
            for problem in st.session_state.sorted_examples:
                problem_id = problem["question"]
                if problem_id in annotations:
                    problem["annotations"] = annotations[problem_id]
                else:
                    problem["annotations"] = {}

            # Initialize examples BEFORE creating tabs
            examples = st.session_state.sorted_examples

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
                search_term = st.text_input(
                    "Search files",
                    "",
                    placeholder="Model name or date...",
                    help="Search by model family, date, or method",
                )

                col1, col2 = st.columns(2)
                with col1:
                    unique_models = sorted(
                        {fd.get("model", "Unknown Model") for fd in file_data},
                        key=lambda x: (
                            not x.startswith("Llama"),  # Llama first
                            not x.startswith("Qwen"),  # Then Qwen
                            x.split()[-1],  # Sort by model size
                        ),
                    )
                    selected_model = st.selectbox(
                        "Model Family", ["All Models"] + unique_models, index=0
                    )
                with col2:
                    unique_dates = sorted(
                        {fd.get("date") for fd in file_data if fd.get("date")},
                        reverse=True,
                    )
                    selected_date = st.selectbox(
                        "Date", ["All Dates"] + unique_dates, index=0
                    )

            st.markdown("---")
            st.markdown("**Selected File**")
            filtered_files = [
                f
                for f in file_data
                if (selected_model == "All Models" or f["model"] == selected_model)
                and (selected_date == "All Dates" or f.get("date") == selected_date)
                and search_term in f["full_name"].lower()
            ]
            selected_file = st.selectbox(
                "Choose results file",
                options=[f["full_name"] for f in filtered_files],
                format_func=lambda x: next(
                    f["display_name"] for f in filtered_files if f["full_name"] == x
                ),
                help="Files sorted by modification date (newest first)",
                label_visibility="collapsed",
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
            st.rerun()  # Force immediate update

    with col2:
        if st.button("Next ➡️"):
            st.session_state.current_idx = min(
                total_examples - 1, st.session_state.current_idx + 1
            )
            st.session_state.input_idx = st.session_state.current_idx
            st.rerun()  # Force immediate update

    with col4:
        st.number_input(
            "Go to",
            min_value=0,
            max_value=total_examples - 1,
            key="input_idx",
            label_visibility="collapsed",
        )
        if st.session_state.input_idx != st.session_state.current_idx:
            new_idx = max(0, min(st.session_state.input_idx, len(examples)-1))
            if new_idx != st.session_state.current_idx:
                st.session_state.current_idx = new_idx
                st.rerun()

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
                    st.markdown(
                        f"```\n{example.get('question', example.get('input_question', 'N/A'))}\n```"
                    )

                    # Display primary response
                    if "response" in example:
                        st.subheader("Model Response")
                        st.code(
                            example["response"], language="markdown", wrap_lines=True
                        )

                    # Display all responses if available
                    if "all_responses" in example:
                        with st.expander(
                            f"View All {len(example['all_responses'])} Responses",
                            expanded=False,
                        ):
                            for i, resp in enumerate(example["all_responses"], 1):
                                st.markdown(f"**Response {i}**")
                                st.code(resp, language="markdown", wrap_lines=True)

                with col2:
                    st.subheader("Answers")
                    answer_col1, answer_col2 = st.columns(2)

                    with answer_col1:
                        correct_answer = (
                            example.get("correct")
                            or example.get("correct_answer")
                            or example.get("target")
                        )
                        st.markdown(f"### Correct Answer")
                        if correct_answer is not None:
                            st.markdown(
                                f"<h2 style='color: #2ecc71;'>{correct_answer}</h2>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.warning("No correct answer found in data")

                    with answer_col2:
                        pred_answer = (
                            example.get("predicted_answer")
                            or example.get("predicted")
                            or "N/A"
                        )
                        is_correct = example.get("is_correct", False)
                        color = "#2ecc71" if is_correct else "#e74c3c"
                        st.markdown(f"### Model Prediction")
                        st.markdown(
                            f"<h2 style='color: {color};'>{pred_answer}</h2>",
                            unsafe_allow_html=True,
                        )

                    if "confidence" in example:
                        st.metric(
                            "Confidence Score",
                            f"{example['confidence']:.2%}",
                            help="Model's self-reported confidence in the answer",
                        )

            with tab2:
                st.subheader("Response Analysis")

                # Show answer distribution
                if "answer_counts" in example:
                    st.markdown("**Answer Distribution**")
                    counts = example["answer_counts"]
                    chart_data = {
                        "Answer": list(counts.keys()),
                        "Count": list(counts.values()),
                    }
                    st.bar_chart(chart_data, x="Answer", y="Count")
                else:
                    st.write("No answer distribution data available")

                # Show confidence metrics
                if "confidence" in example:
                    st.markdown(f"**Confidence Score**: {example['confidence']:.2%}")

                # Show majority voting info
                if "majority_answer" in example:
                    st.markdown(f"**Majority Answer**: {example['majority_answer']}")

                # Show self-consistency metrics
                if "all_responses" in example:
                    total_responses = len(example["all_responses"])
                    correct_responses = sum(
                        1
                        for resp in example["all_responses"]
                        if abs(extract_answer(resp) - example["correct"]) < 1e-6
                    )
                    st.markdown(
                        f"**Self-Consistency**: {correct_responses}/{total_responses} "
                        f"({correct_responses/total_responses:.2%})"
                    )

                # Show error types if available
                if "error_types" in example:
                    st.subheader("Error Categories")
                    for error in example["error_types"]:
                        st.markdown(f"- {error}")
                else:
                    st.write("No detailed error categorization available")

            with tab3:
                st.json(example)


if __name__ == "__main__":
    main()
