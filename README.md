## LLM reasoning

Currently Refactoring

### Results on GSM8K

#### Links

- [llama3.2 benchmarks blog](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
- [qwen2.5 benchmarks blog](https://qwenlm.github.io/blog/qwen2.5-llm/#qwen25-7b-performance)

#### Zero-shot CoT

`Llama-3.2-1B-Instruct` - GSM8K - 38.29% Accuracy (44.4 reported 8-shot CoT) - 46.55%

`Llama-3.2-3B-Instruct` - GSM8K - 69.45% Accuracy (77.7 reported 8-shot CoT) - 79.61%

`Llama-3.1-8B-Instruct` - GSM8K - 79.61% Accuracy (84.5 reported 8-shot CoT) - 85.97%

`Qwen2.5-7B-Instruct` - GSM8K - 87.34% Accuracy (85.4 reported 4-shot CoT)

#### Few-shot CoT (8-shot)

pre-prompt fix - post fix (fixed + zero-shot instruction prompt)

`Llama-3.2-1B-Instruct` - GSM8K - 40.64% Accuracy (44.4 reported 8-shot CoT) - 44.88% (45.79%)

`Llama-3.2-3B-Instruct` - GSM8K - 73.54% Accuracy (77.7 reported 8-shot CoT) - 78.62% (79.00%)

`Llama-3.1-8B-Instruct` - GSM8K - 82.64% Accuracy (84.5 reported 8-shot CoT) - 84.76% (84.76%)

`Qwen2.5-7B-Instruct` - GSM8K - 85.44% Accuracy (85.4 reported 4-shot CoT)
