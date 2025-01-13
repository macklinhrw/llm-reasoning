---
date: 2025-01-07
tags:
  - unique-note
  - projects/llm-reasoning
aliases:
---
related:: -
- [[2025-01-07]]
- [[202412191528 - week 5 meeting notes]]
- [[202412190331 - types of errors]]
- [[202501012041 - plan for tasks week 7 and failure modes]]
- [[202501052039 - new error categories]]
- [[202412301516 - meeting week 6]]
- [[202412242249 - reasoning reflection on paper till week 6]]
- [[202410230144 - cs224N - lecture 8]]
- [[202411031244 - Complementary Explanations for Effective In-Context Learning|Complementary Explanations for ICL]]
- [[202501072022 - meeting week 7]]
- [[202411250124 - From Decoding to Meta-Generation- Inference-time Algorithms for Large Language Models]]
- [[202411281911 - Unlocking the Capabilities of Thought- A Reasoning Boundary Framework to Quantify and Optimize Chain-of-Thought|Unlocking the Capabilities of Thought]]
- [[202501120746 - At what point are problems too difficult for LLMs and can we model this?]]
- [[Brainstorming - at what point are problems too difficult - 2025-01-11 18.51.29.excalidraw]]
- [[verify-step-by-step.pdf]]
- [[202411081022 - Towards Understanding Chain-of-Thought Prompting - An Empirical Study of What Matters]]


# 202501071248
---

## Outline of my exploration
- First, if I want to do research, ask what is research?
	- Expanding the frontier understanding or knowledge about a particular topic.
- Choosing a research question
	- Generally determining an area I'm interested in, to focus the following exploration.
		- Chain of Thought (reasoning)
	- What do we know?
	- What do we not know?
		- I explored this question since to answer it you also need to answer what we know.
	- Once I have found an (important/significant) area we don't know that looks both promising and interesting, I can design a research question to specifically target it.
		- What are the precise definitions of the vocabulary in the question?
		- What is the scope of the question?
		- Will it be impactful/significant if answered?
		- Is it able to be answered?
			- (with an ideal experiment with no lack of resources and time)
		- Am I able to answer it (with my limited resources)?
			- (with an experiment I can execute)
- Designing an experiment
	- "experiment" section in research papers, I have seen it plenty times before
	- How will the experiment contribute to answering the research question?
		- follow logically step by step to reach a conclusion (answer to the research question)
	- What is the ideal experiment that will answer the question?
		- (no lack of resources and time)
	- What is a realistic experiment?
		- (with an experiment I can execute)
- Conclusion

## Research Questions
### [[202501120746 - At what point are problems too difficult for LLMs and can we model this?|At what point are problems too difficult for LLMs and can we model this?]]
- "problems"
	- Specifically focusing on mathematics reasoning problems
- "difficulty"
	- [[202412242249 - reasoning reflection on paper till week 6#^3e902f]]
	- ![[202411281911 - Unlocking the Capabilities of Thought- A Reasoning Boundary Framework to Quantify and Optimize Chain-of-Thought#^524397|Unlocking the Capabilities of Thought]]
	- mention: [[202411081022 - Towards Understanding Chain-of-Thought Prompting - An Empirical Study of What Matters|Understanding CoT - Empirical Study]] (part on scaling laws)
	- what are the "number of reasoning steps"?
		- Different number of independent calculations, sub-problems needed to arrive at the final solution
		- Example (quickly made up): x = y + z, we know z = 10, and y is the number of cats and dogs owned by person K, he owns 5 cats and 2 dogs.
			- 1. y = 5 + 2 = 7
			- 2. x = 10 + 7 = 17
			- final answer: 17
		- Requires 2 reasoning steps. I haven't referred to the papers, just using this as a working definition.
	- what is "computational complexity"?
		- This seems a little vague to me, and it might refer to the difficulty of the mathematical calculation(s) needed to answer a question. I would need to refer to the paper to have a better idea for what the author intended.
	- Answering exactly what difficulty it is can be complex, because there are different components to difficulty
		- Components: common sense, planning, etc.
			- What labels are important?
		- Specifically in GSM questions, where does the difficulty come from?
		- I don't know, this can be varied throughout the dataset, so I would need evaluate them myself because no labels exist already
		- There might be some automatic evaluation technique possible? Could be explored.
		- We might be able to simplify this problem with a heuristic, like just looking at the number of reasoning steps.
			- This is similar to what the [[202411281911 - Unlocking the Capabilities of Thought- A Reasoning Boundary Framework to Quantify and Optimize Chain-of-Thought|Unlocking the Capabilities of Thought]] did.
				- Also looked at size of numbers in arithmetic calculations.
			- But if we do this, it doesn't seem to go past what was already done in that paper.
 - "too difficult"
	- The model can't get the problem right using greedy decoding?
	- Or, can the model not get the problem correct after infinite k-shot attempts?
	- I think both are interesting to address, the latter question might be too difficult. We can't really answer it without knowing something more fundamental about the model because we can't experimentally determine this (can't run infinite generations).
- "at what point"
	- basically, knowing the difficulty threshold for when a model can no longer solve a problem
	- is this even possible?
		- we can't run an evaluation to test this, and I don't know if there is a good heuristic
- "can we model this"
	- does this mean some sort of predictor?
	- I think a predictor is the best way
- "LLMs"
	- Transformers, but after analyzing the question more this might be too general if we are to train a predictor because we can't determine dataset quality which is a confounding variable
- What experiment, given difficulty labels, can we design to answer the research question?
	- Overall this comes down to finding some sort of correlation between difficulty and an LLM's ability to solve a GSM problem.
	- Predictor?
		- *The variables*: Model size, difficulty, solved or not solved
		- A potentially better approach than looking at solved or unsolved is looking at the number of reasoning steps that were solved correctly, but this could require some extra work and complex parsing
		- For the experiment "can we model this": train a predictor to predict chance of correctness based on a set of features: model size, train dataset size, 
		- I don't think having a model that predicts purely based on model size will be useful because it will quickly become inaccurate due to differences of dataset quality and dataset size.
		- If I test the model only on llama models it may be accurate because dataset quality and size may be similar
			- (llama3: 15T tokens, 15.6 for 405B model; use classifiers and heuristics for data quality. qwen2.5: 18T tokens, just mentions "sophisticated filtering and scoring mechanisms, combined with strategic data mixture")
		- Then, after training a predictor, is it able to accurately predict, based on difficulty labels and *other features*, whether the model will accurately get a question correct?
			- other features: model size
		- Question: different models are trained differently, this means their accuracy will vary (potentially by a lot) even in the same model size, how can we accurately model it due to the confounding variables (that we may not know) like dataset size and dataset quality?
		- dataset size may be able to be known on certain models which released the details in the technical report
		- for dataset quality, we have no way of knowing
		- Assume we are able to come up with meaningful labels that accurately describe the difficulty of problems in the GSM dataset, now, what can we do with these labels to answer the research question?
		- Question: Assume this predictor works and can be fairly accurate overall when judging based on variables of dataset size and model size, is this useful and does it answer our question?
			- We can graph the predicted accuracy on varying degrees of difficulty
			- How do these graphs vary based on model size, dataset size?
	- Threshold?
		- ...
- ...

### *How are reasoning paths represented in LLMs and how are they chosen?*
- "Reasoning path" - Referring to a CoT output, but multiple different outputs can be the same reasoning path: They may have different words but the same essential components of the reasoning are the same.
	- "essential components" - basing this on the week 1 papers, it is both the *computational trace* (numbers, mathematic operations), and *language*. If the language conveys the same meaning, and the computational trace is effectively the same. The result is also an essential component. If any of these essential components are different, then I would consider the reasoning path to be *different*.
		- [[202411031244 - Complementary Explanations for Effective In-Context Learning|Complementary Explanations for ICL]]
- "chosen" - As in samples via. a decoding algorithm from a softmax distribution.
	- I don't think I can answer why it is chosen?
	- We can certainly look at the probability of some number of top-k options
- Seems very difficult to answer
- This question is still interesting, maybe modify it then explore a simplified version?
- Is there a reasoning that can get a problem right (maybe not the greedy decoded one)?

### How do we characterize the outputs of CoT?
- This question is pretty general - how do I explain each part of this, and concretely identify what it means and how to explore it? 
- Specifically (and also simply) can I state what it means, at least to the research I would like to do
- Word definitions
	- Characterize - "describe the distinct nature or features of"
	- Outputs - The answers, but does it also include the reasoning chain? "outputs of CoT", does this mean the final answer of the reasoning chain?

## What to do
- Then after I explore this a little more, how can I think of an experiment (both ideal and realistic versions) that would answer that question.
- I can write down my thoughts as a complete chain, first with a basis then going through each step, including experiment, till I arrive at a conclusion that is logically sound. Does each step make sense and support the next?
- For the question(s) I choose to explore, can I explain the precise scope? What does it include and what does it not include.

*Taking a step back* from the specific questions and scenarios we have talked about.

There are a couple questions (big picture):
## Questions
- What do we know?
- What do we not know?
- Out of the things that we don't know
	- What is worth exploring (and will have significant results/impact)
	- What is able to be explored (with a well-designed experiment)

## What do we know?
- Looking at the first week, we seem to know that CoT is robust to exemplars, but language and computation traces are important together
	- [[202411081022 - Towards Understanding Chain-of-Thought Prompting - An Empirical Study of What Matters|Understanding CoT - Empirical Study]]
- (exploring this below instead)

What we don't know is linked to what we know, which I needed some context on to answer. I ended up including the answers to this question below. I think looking through the perspective of what we don't know is very useful.
## What do we not know
- Can LLMs reason and to what extent?
	- Hard to answer, which makes me want to explore something easier and that has a definite way of answering. Reasoning could be subjectively defined, it might have different definitions which are constantly changing (or getting upgraded). There might not be much consensus on this. (This would need more exploration.)
- How and why do LLMs decide on specific reasoning paths?
	- Also difficult to answer. I'm not sure how to go about answering it.
	- I think this is interesting, and there hasn't been work that I'm aware of specifically focusing on this.
- When will LLMs (*Transfomers*) stop improving performance-wise using the approach of more data, more training time?
	- I don't believe this can be answered directly because there are sub-questions and elements that too general/broad in the question.
	- What is meant by performance?
	- More data? What kind of data?
	- More training time? Training hyper parameters have an effect.
	- Implicitly I was thinking about the Transformer architecture (*maybe the question should be changed*)-- but does it also encompass different modifications, for example there are different types of positional embedding techniques? Which is it referring to? Different architectures?
		- If its general its hard to answer specifically.
	- **I'm not interested in answering this, I'm just trying to write down everything that I can think of (and for practice).**
- Chain of Thought
	- **Why does CoT work?**
		- We know some of the components of how it works, but we don't know exactly why it works except we have some idea that the Transformer must learn it (unsupervised) from the training data.
		- (question: does fine-tuning improve CoT performance and why? From what I've seen, my hypothesis is that it is improved. I would need to verify this. *Is this worth answering and why*?)
	- **How does CoT work?**
		- This is what I have seen explored in the papers from week 1, but what is unexplored?
			- One thing: we have some idea of how exemplars impact performance of few-shot CoT, but what are the elements that might impact zero-shot CoT (this means modifications to the prompt)?
				- [[202411031239 - Large Language Models are Zero-Shot Reasoners|Zero-Shot CoT]]
				- i.e., is just using some template with "Think step by step" the best?
				- I think there must be some papers exploring this, but I haven't read them or searched for them
				- how fragile is the model towards changes in its zero-shot prompt?
		- papers from week 1
			- [[202411081022 - Towards Understanding Chain-of-Thought Prompting - An Empirical Study of What Matters|Understanding CoT - Empirical Study]]
			- [[202411061946 - Text and Patterns - For Effective Chain of Thought It Takes Two to Tango|For Effective Chain of Thought It Takes Two to Tango]]
			- [[202411031244 - Complementary Explanations for Effective In-Context Learning|Complementary Explanations for ICL]]
	- **To what extent does CoT work?**
		- I've read that it fails in some of the same ways that humans would fail in tasks where overthinking has a negative impact.
			- [[202411092138 - Mind Your Step (by Step) - Chain-of-Thought Can Reduce Performance on Tasks Where Thinking Makes Humans Worse|Mind Your Step]]
		- Restores scaling curve in tasks direct prompting performs poorly (with flat scaling) on, such as scientific and mathematical reasoning questions (i.e., grade school math).
			- [[202411011536 - Chain-of-Thought Prompting Elicits Reasoning in Large Language Models|Few-Shot CoT]], [[202411031239 - Large Language Models are Zero-Shot Reasoners|Zero-Shot CoT]]
		- After a certain point of ==difficulty==, or on out of distribution problems, it will perform poorly.
			- [[202411121939 - Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks|Reasoning or Reciting?]]
		- So what we don't know is, what factors go into a model being able to solve a specific difficulty of problem? (Can look at scaling curve for model sizes as well.)
		- Does the model know what it doesn't know?
		- If a model doesn't know how to answer, how likely is it to hallucinate?
		- To what extent is it considering different reasoning paths at each token? To what degree is the model thinking about different ways to solve the problem.
			- This is important because more ==complex== problems may require more exploration
			- "complex problems" - what kind of complexity?
			- If a question requires the model to traverse a graph, and this graph is complex, how does the model decide which nodes to explore? Can it explore nodes in parallel?
				- [[202412212022 - Forking Paths in Neural Text Generation]]
			- ...
		- If a problem is very difficult, and useful to break it down into simpler subproblems, is the CoT prompted model able to solve it? Does it represent these sub problems or does it solve the problem in a different way? (Maybe the problem doesn't require subproblems, but humans would solve it using subproblems because the direct solution is too difficult-- this is a problem solving strategy I've seen applied to competition-level math problems.)
			- "useful to break it down into simpler subproblems" - do these labels exist? can we judge this? is there a dataset?
			- the *least to most prompting* paper -- for example does an explicit strategy for breaking down subproblems through prompting, or a pipeline approach, allow models to solve problems they weren't able to solve before with regular CoT
			- it would be significant to know pipelines or prompting strategies that are more effective than CoT at problem solving, so we can use LLMs to solve problems that are more difficult
			- **I think all this simplifies as "is CoT able to sufficiently break down long complicated problems (requiring planning) and solve them more successfully than other techniques?"**
				- [[202411140911 - Tree of Thoughts - Deliberate Problem Solving with Large Language Models|Tree of Thoughts]]
				- ToT isn't tested on GSM8K (mathematical reasoning)
				- [[202412141300 - Self-Refine- Iterative Refinement with Self-Feedback|Self-Refine]] didn't improve results on GSM8K
				- 
		- 


Thought: can research be thought of as exploring a graph of questions? I ended up organizing my above notes this way which seems useful.

## My thoughts on exploring the extent of CoT capabilities versus how it works
- "Extent of CoT capabilities" - When does CoT fail to solve problems
	- Another question and natural extension is "how to improve it" or "how to use it more effectively"
- Thinking about it some more, I think its conditional how actionable the answers to these questions can be. It depends on the specifics, and answering both can be interesting and actionable, though in different ways.

## Actionable versus just understanding
- Why think about this: I want to understand what kind of research I want to do. Answer the question, is having an actionable paper important? Intuitively I would think so because it would have a greater impact where people can act on the results of your research.
- Yet if your paper deepens the fields understanding on a concept (crucial to that field), it can also be significant.
- Are these both the same thing? Is a paper that deepens understanding inherently actionable?
- Understanding can guide future actions, for example if we know (i.e., [[202411092138 - Mind Your Step (by Step) - Chain-of-Thought Can Reduce Performance on Tasks Where Thinking Makes Humans Worse|Mind Your Step]]) what a model architecture is bad at (or a prompting technique - CoT), when we previously didn't, it can give researchers the idea to focus on the strengths and come up with techniques to avoid the weaknesses, or researchers can explore methods to modify the architecture to fix the weakness.
- An example of a directly actionable paper is [[202411281911 - Unlocking the Capabilities of Thought- A Reasoning Boundary Framework to Quantify and Optimize Chain-of-Thought|Reasoning Boundary Framework]], it first deepens our understanding by analyzing reasoning boundaries (an upper bound for CoT), then talks about moving actions (such as arithmetic with large numbers) into the reasoning boundary (decomposing computations into smaller numbers for example) to increase performance. The paper focuses on understanding, but also has a directly actionable component: a method to improve performance of CoT.
- 

## End accuracy versus step-wise accuracy
- Exploring potential correlations to step-wise accuracy could be more meaningful than only exploring ones in relation to end-accuracy.
- correlations
	- difficulty

## The difficulty with difficulty labeling
- specifically with diverse and a larger number of labels
- hypothesis: inherently most problems will have different sources of difficulty
- if they are mislabeled, the data will not be complete and give inaccurate results or correlations 
	- i.e., finding correlation of commonsense difficulty with end-accuracy
	- if some of that difficulty comes from another type, then the results would be inaccurate and the difficulty unaccounted for could be faultily attributed to just commonsense (if its the only one labeled)