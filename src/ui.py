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


@st.cache_resource
def get_result_files(directory="results/incorrect"):
    """Get all available result files."""
    file_patterns = [
        os.path.join(directory, "*.json"),
        os.path.join(directory, "*.jsonl"),
    ]
    files = []
    for pattern in file_patterns:
        files.extend(glob.glob(pattern))

    # Sort files by modification time (newest first)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # Create friendly names for display
    file_options = {os.path.relpath(f, directory): f for f in files}
    return file_options


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


def main():
    st.set_page_config(page_title="Dataset Viewer", page_icon="🔢", layout="wide")
    st.title("Dataset Viewer")

    # Data source selection
    data_source = st.sidebar.selectbox(
        "Choose data source", ["GSM8K Dataset", "Model Results"]
    )

    if data_source == "GSM8K Dataset":
        with st.spinner("Loading dataset from Hugging Face..."):
            dataset = load_gsm8k_data()

        if dataset is None:
            st.warning("Failed to load dataset.")
            return

        split = st.sidebar.selectbox("Choose dataset split", ["train", "test"])
        examples = dataset[split]

    else:  # Model Results
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

    # Initialize session state
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
        st.session_state.input_idx = 0

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

    current_idx = st.session_state.current_idx

    # Display progress
    st.progress(current_idx / (total_examples - 1))
    st.markdown(f"**Example {current_idx + 1} of {total_examples}**")

    # Display current example
    if 0 <= current_idx < total_examples:
        example = examples[current_idx]

        # Display content based on data source
        if data_source == "GSM8K Dataset":
            st.subheader("Problem")
            st.code(example["question"], language=None)
            st.subheader("Solution")
            st.code(example["answer"], language=None)
        else:
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
