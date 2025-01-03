import streamlit as st
from datasets import load_dataset
import random


@st.cache_resource
def load_gsm8k_data():
    """Load GSM8K dataset from Hugging Face."""
    try:
        # Load both splits at once
        dataset = load_dataset("gsm8k", "main")
        # Convert to lists for faster access
        return {"train": list(dataset["train"]), "test": list(dataset["test"])}
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None


def main():
    st.set_page_config(page_title="Dataset Viewer", page_icon="🔢", layout="wide")

    st.title("Dataset Viewer")

    # Load entire dataset
    with st.spinner("Loading dataset from Hugging Face..."):
        dataset = load_gsm8k_data()

    if dataset is None:
        st.warning("Failed to load dataset.")
        return

    # Dataset split selection
    split = st.sidebar.selectbox("Choose dataset split", ["train", "test"])
    examples = dataset[split]
    total_examples = len(examples)

    st.sidebar.markdown(f"Total examples: {total_examples}")

    # Initialize both states if they don't exist
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
        st.session_state.input_idx = 0  # Initialize input_idx along with current_idx

    # Navigation controls
    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

    with col1:
        if st.button("⬅️ Prev"):
            st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
            st.session_state.input_idx = (
                st.session_state.current_idx
            )  # Keep input in sync

    with col2:
        if st.button("Next ➡️"):
            st.session_state.current_idx = min(
                total_examples - 1, st.session_state.current_idx + 1
            )
            st.session_state.input_idx = (
                st.session_state.current_idx
            )  # Keep input in sync

    with col4:
        # Direct number input for jumping to specific examples
        st.number_input(
            "Go to",
            min_value=0,
            max_value=total_examples - 1,
            key="input_idx",  # This will automatically update st.session_state.input_idx
            label_visibility="collapsed",
        )
        # Only update current_idx if input_idx has changed
        if st.session_state.input_idx != st.session_state.current_idx:
            st.session_state.current_idx = st.session_state.input_idx

    current_idx = st.session_state.current_idx

    # Display progress
    st.progress(current_idx / (total_examples - 1))
    st.markdown(f"**Example {current_idx + 1} of {total_examples}**")

    # Display current example
    if 0 <= current_idx < total_examples:
        example = examples[current_idx]

        st.subheader("Problem")
        st.code(example["question"], language=None, wrap_lines=True)

        st.subheader("Solution")
        st.code(example["answer"], language=None, wrap_lines=True)

        # Display metadata
        with st.expander("Metadata"):
            st.json(example)


if __name__ == "__main__":
    main()
