## LLM reasoning

Currently Refactoring

### Results on GSM8K

#### Links

- [llama3.2 benchmarks blog](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
- [qwen2.5 benchmarks blog](https://qwenlm.github.io/blog/qwen2.5-llm/#qwen25-7b-performance)

#### Zero-shot CoT

`Llama-3.2-1B-Instruct` - GSM8K - 46.55% Accuracy (44.4 official reported 8-shot CoT)

`Llama-3.2-3B-Instruct` - GSM8K - 79.61% Accuracy (77.7 official reported 8-shot CoT)

`Llama-3.1-8B-Instruct` - GSM8K - 85.97% Accuracy (84.5 official reported 8-shot CoT)

`Qwen2.5-7B-Instruct` - GSM8K - 87.34% Accuracy (85.4 official reported 4-shot CoT)

#### Few-shot CoT (8-shot)

score (score + zero-shot instruction prompt)

`Llama-3.2-1B-Instruct` - GSM8K - 44.88% Accuracy (44.4 official reported 8-shot CoT) (45.79%)

`Llama-3.2-3B-Instruct` - GSM8K - 78.62% Accuracy (77.7 official reported 8-shot CoT) (79.00%)

`Llama-3.1-8B-Instruct` - GSM8K - 84.76% Accuracy (84.5 official reported 8-shot CoT) (84.76%)

`Qwen2.5-7B-Instruct` - GSM8K - 85.44% Accuracy (85.4 official reported 4-shot CoT)

### Optional Setup

#### NLP tool setup (for refine eval)

`python -m spacy download en_core_web_sm`

```python
>>> import nltk
>>> nltk.download('punkt_tab')
```
