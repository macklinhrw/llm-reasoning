---
date: 2025-01-20
tags:
  - unique-note
  - projects/llm-reasoning
aliases:
---
related:: 
- [[2025-01-20]]
- [[202501120746 - At what point are problems too difficult for LLMs and can we model this? - Part 1]]
- [[Difficulty to error types 2025-01-13 12.03.44.excalidraw]]
- [[202501201224 - week 8 meeting notes]]
- [[202501071248 - week 8 - what do I research?]]

# 202501201131
---
## Question and decomposition
- At what point are problems too difficult for LLMs *and can we model this*?
- Decomposition
	- "difficulty" - Consider two problems, one where the model gets the answer correct and another where the model gets it incorrect. My definition of difficulty is the differentiating factor between these two problems that allows 1) the model to answer the former question correctly and causes 2) the model to get the latter question incorrect. Difficulty comes from, I hypothesize, several discrete sources that range in how they impact the problem solver's ability to solve the problem successfully.
	- "problems" - Mathematical Reasoning problems, specifically we have been looking at GSM8K. I can see expanding this range to other datasets if we are looking for problems with more diverse sources of difficulty.
	- "LLMs" - Specifically, looking at specific size classes of models should be useful because we can evaluate based on this variable of size, how does it affect the model's ability to generate accurate results.
	- "and can we model this?" - This is a separate (but linked) question, I could remove it from the question. For now I will save it for later to think about.

## More In-depth exploration of the definition of difficulty
- Indivisible versus compound sources
- I hypothesize that there will be one or multiple indivisible and discrete sources for difficulty in any particular mathematical reasoning problem.

## What do I mean by Difficulty Type and Error Type
- Difficulty Type
	- Indivisible (cannot be further divided into component difficulty types) source of difficulty in any particular problem
- Error Type
	- Fundamentally an error type (or multiple?) is the direct reason for why a model failed to get a problem correct. 
		- Correctness could be determined by **end accuracy** for a discrete (0, 1) value. It could also be continuous based on **step-wise accuracy** (0-1, percentage of steps correct).
		- Do we only care about the end answer being correct? Or do we care about how many steps the model got correct?
	- A type of error in a generation for a problem that a model may have made due to difficulty, but not definitely because of that difficulty. There could be more variables at play to why a model generates an error that are not necessarily linked to the inherent difficulty of that problem.
	- For example, the model may output the answer in a way that is not expected by the parser, resulting in lower accuracy. This is not dependent on the difficulty of the problem (though it may have some small correlation).
		- Think about this scenario: A problem is difficult in some nuanced way that the model is not good at. The model is "uncertain"* about the best reasoning path, so many tokens that lead to different reasoning paths and end-answers are more likely. Let's say in the greedy decoding scenario, the model keeps generating tokens that steer it in a particular direction, and because the model is conditioned on its previous outputs the model's end answer could change drastically based on it having made some sort of (uncertain) decision earlier in the generation. As a result the error might stem from that token which is irrelevant to the reasoning but due to the _model internals_,  will cause a change in the end-answer.
			- This is **not related to finding error types** (for finding difficulty types), I think these should be clearer because when a model makes a mistake, then the mistake will be visible, but the cause of that mistake can sometimes very difficult trace due to a scenario like this.
			- When the model is uncertain it will have a wider spread of answers with more generations (at a higher temperature value).
			- Question: What was the direct cause of this error?
				- The answer is hard to trace
			-  ![[Pasted image 20250127100412.png]] (page 7, [[Forking Paths in Neural Text Generation.pdf]])
			- [[202412212022 - Forking Paths in Neural Text Generation]]
	- Error Types are not causally linked with difficulty.
	- For any problem, however, and over a large amount of generations, it should be possible to identify correlations between difficulty and error types.
	- Overall, difficulty may be correlated with error types in model generations, and it may be possible to trace these correlations back to their root difficulty source in the conditioned problem.
- Commonsense? Both?
	- Is it a difficulty type or an error type?
	- A difficulty type, it might be easy to associate with an error-- but it is not an *error*, it is something inherent to the question. The error might be the result of the model lacking commonsense, resulting in the error.
	- Specifically, the error should be "Lacking Commonsense Knowledge", and the difficulty type would be "Commonsense Knowledge". Both are closely related, before the error type label "Commonsense" wasn't specific enough resulting in some confusion

## Related questions
- Does a model know what it can and can't do?
- Is model uncertainty similar to an internal representation of question difficulty?

## Defining reasoning paths (simplistic definition)
- I would define reasoning paths as being different generations by a model with distinct reasoning components
	- Reasoning Components
		- Natural Language
		- Computational Trace
		- [[202411031244 - Complementary Explanations for Effective In-Context Learning|Complementary Explanations for ICL]]
	- "distinct"
		- I can't yet think of any good way to directly measure this, particularly in the case where the model result is the same over different paths, how do we know if the reasoning paths are distinct?
		- It is easier to look at the end answer, and if its different we know for certain there are different reasoning paths, because with the same reasoning the answer will be the same (except in cases where the answer doesn't follow the reasoning steps; i.e., the model comes up with a completely different answer than what was arrived at by the CoT--  I don't know how likely this is, and experimentally I haven't seen it happen yet)
		- It is much more difficult to look at the components, natural language and computational trace, and come up with a way to tell if these are distinct.
- This is not well-defined in the literature I have read, and may just be used loosely as a term to refer to different CoT generations*
	- [[202411142014 - Self-Consistency Improves Chain of Thought Reasoning in Language Models|Self-Consistency]]

## Experiment
^38cc20

- Particular model and K-value (number of generations)
- Temperature value `0.7`
- Take a dataset (GSM8K)
- Generate responses
	- difficulty metric - percent of these generations that are correct
- After
	- Filter them for more difficult problems
	- Look at what makes them difficult: I can identify difficulty type

## Difficulty metric
- Percent of K-generations gotten correct (end-answer)
- Reworded, this is the same as the chance the model will get a problem correct (given K-generations), determined experimentally.

## Looking at Difficulty Type
- If I identify some sort of type of difficulty, how do I know that this is a correct label (and the best possible)?
	- Indivisbility (unable to be decomposed/factored into other difficulty types)
	- Applicable to a wide range of problems
		- This may not necessarily be a property of a good label. Will some problems have difficulty that is extremely-rare or even unique to that particular problem?
		- A comparison that comes to mind is the periodic table.
	- I don't think its possible to directly identify what are the best possible labels, I can only refine them after lots of thinking which is part of the process.

## Experiment results
- A simple analysis that can be done, given that I'm able to (accurately) label difficulty types in problems, is to graph these types and their relation to end-accuracy. Is it useful?
- A question that my previous experiment tried to answer (maybe unsuccessfully) was the objective _degree_ of difficulty each difficulty type had in a particular problem.
	- Moving away from model-specific or K-value specific.
	- Can we determine a generalizable difficulty metric.
- Will looking at step-wise accuracy allow us to get a more nuanced view of difficulty?
	- Instead of taking percentage of problems solved, take one minus the average percentage of steps solved accurately as the difficulty metric (given specific model and K-value).
	- I'm curious how these would differ: what problems previously filtered out may now be included in the top X, etc.
- Now that I have these difficulty labels, what do I do with them?
	- Analyze/label degree of difficulty
	- By analyzing thresholds
- How useful is just having these difficulty labels? (maybe for multiple models, looking at multiple K values)
	- Still useful, but maybe to a smaller extent depending on comprehensiveness

## Difficult versus easy
- "Easy" is also a classification of difficulty, so why just look at "difficult" questions?
	- We want to know what the model is incapable of
	- Difficulty that leads to errors and the model getting the end-answer incorrect
	- The difficulty in this context is "what causes the model to get the answer incorrect". Is this a good way to look at difficulty?
		- Difficulty is inherently related to accuracy by its *definition*: it should have some negative correlation (looking at an overall difficulty measure, which is inherently compound, this may not always be true, but broken down into indivisible components I hypothesize it will always be true; models may good at handling different types of difficulty better, so overall it may look like this doesn't hold but when breaking it down into types of difficulty, it should hold). 
		- This is a good way to start looking at difficulty.
	- If we think about this scenario where the model gets a problem incorrect, and there is some hypothetical number that perfectly represents the difficulty of a problem, there may be a threshold value after which the model gets all problems incorrect. (greater than or equal to a threshold value)
- Difficulty is continuous

## Possible difficulty types (brainstorming)
- Minimum required reasoning steps
- Arithmetic complexity (size of numbers in calculations)
- Algebraic complexity
	- number of variables in calculations
- Specific problem solving knowledge
- Common sense

## Difficulty Types (experimental -- being updated)
- Ambiguous
	- What the question is asking for has multiple interpretations. The model might tend to interpret it one way while the target solution does another resulting in inaccuracy in matching the solution's answer.
- Output Format
	- Similar to being ambiguous but more specific. The question isn't clear on the exact answer format. The solution may do it one way while the model may do it another.
- Unneeded Information
	- Extra information that is not needed to solve the problem is included in the question. The difficulty here comes from understanding it is not needed for calculations. A model may fail here for example is if it uses this information in calculations that impact its final answer in a negative way.
	- Other ways to think about it
		- Experiment: Model initially gets the problem incorrect, after removing the unneeded information the model is more likely to get it correct (i.e., in K generations)-- though this might have edge cases
		- Information included in the problem that is not used in any (minimal) correct solution path.
			- minimal correct solution path - no extra information that is not used in reasoning steps that make progress towards a final answer
- Commonsense Knowledge
	- Outside knowledge is needed to understand the questions. For example a football question may assume you know the rules of football.
- Information Amount
	- The information amount is large
		- Multiple entities, variables, relations, etc. needed to be able to solve the problem.
		- A resulting error is "forgetting" previous information needed to solve the problem which may be (I hypothesize) due to the model not being able to represent this much information internally.
- Arithmetic Complexity
	- The questions uses large numbers that the model is unable to properly handle when doing arithmetic calculations (wrong answers from arithmetic calculations).
	- Seems to result in a large spread of answers (due to model uncertainty?)
- Not actual difficulty types (useful labels for future dataset filtering, etc.)
	- Incorrect Solution
	- Parser Error
- Other

## Looking at reasoning steps
- What is a reasoning step
	- Combination of natural language and computation trace that makes progress towards an answer given a prompted problem.
	- Not divisible into more reasoning steps.
	- Could be a sentence, a paragraph, as long as its not divisible further.
	- Why not consider divisible reasoning steps?
		- We want it to be the smallest possible unit so when we ask how many reasoning steps are required for a problem we don't get inconsistent answers.


---
Broke the next question down better below.
## ~~Analyzing the question, what experiments would answer it?~~
- *At what point are problems too difficult for LLMs?*
- With the experiment described above ([[#^38cc20]]) I can get a feel for difficulty, its effect on the model, and identify difficulty type labels.
- Looking at the difficulty metric
	- It is a **constrained** difficulty metric
	- Constrained by
		- The model (consider size, model family)
		- The number of generations (K)
	- Naturally I thought about how to go about making an **unconstrained** difficulty metric
	- Difficulty types are specific to a problem and not specific to a model/model generations, so inherently they are generalizable.
	- How could we remove the constraints?
		- It could be easier to *loosen the constraints*, for example instead of being constrained to a specific model, we could do a larger experiment where we evaluate a range of model sizes and model families then look at the general trends in this combined data rather than looking at model-specific results only (which is less useful because we want to know about LLMs in general not a specific LLM).
		- The **number of generations** I think is a reasonable constraint to have, though I think its a good idea to do a small scale experiment and test how changing to K value to be larger will affect the end results
			- For example, lets say we fit a line when graphing **difficulty** vs. **end-accuracy**, how does the line change based on K value, and due to time/budget limitations we could test lets some reasonable subset of the dataset instead of the entire dataset. Say, the first couple hundred samples.
				- Difficulty is already tied to accuracy based on the current metric...
				- Top-K Accuracy - percentage of K-generations for a problem that are correct
				- End-accuracy - 
				- I.e., if none of the generations are correct, accuracy is 0%, difficulty is 100%. If one of the generations is correct (out of K), then the accuracy $\frac{1}{K}$, difficulty is $1 - \frac{1}{K}$.
				- Graphing these two against each other is not meaningful because of their definitions.
		- Based on this small-scale experiment, if the end results aren't affected we can keep using our K value, or if they are we can account for this by finding a better K value, or if this is infeasible (K is too large - time/budget) then we can state the limitations properly

## Difficulty -> Overall experiment
- *At what point are problems too difficult for LLMs?*
- To answer this should require a ==series of experiments==, not just one. If this scope is too large, then we would need to change the question, and could use intermediate results (i.e., the first experiment) to answer it. Still, the end goal could be to answer this more difficult question (with follow-up papers if needed).
- **First Experiment**: We use a *constrained* difficulty metric to identify difficult problems (relatively with these constraints).
	- Constraints: Model, K-generations
	- We can see the problem, we can see the solution, we can see the error the model is making -> this information can help us identify what characterizes the difficulty of this question -> identify difficulty type labels
	- Now we have difficulty labels that are not specific to the model generations, but specific to the question. They are not subject to the constraints, the labeling process was about identifying these.
- **Second Experiment**: We can analyze each label and create a specific scale for the difficulty type to be able classify to what extent, given a particular problem, that this label is contributing to the problem's overall (unconstrained) difficulty.
	- Create the ==difficulty scales== for each difficulty type.
		- This might prove difficult, for example in the "Commonsense Knowledge" difficulty type, how do we determine how common commonsense is? At first thought we could analyze a large corpus for how often this knowledge appears in it, but we would need a method for determining if its present first (even given different language), basically some way to classify if its present or not (train a classifier model?). Then, we use its appearance frequency as the continuous scale on which we rate the difficulty of the "Commonsense Knowledge" label.
			- Difficulty scale - some criteria corresponding to classifying the difficulty of a problem into discrete labels (numbers) in some range (specific to the difficulty type/label). Could also be a continuous scale if it suits the difficulty type better.
		- The above process would require a lot of time and effort.
		- We could consider only looking at a subset of labels of difficulty from the previous experiment if we face limitations.
	- Then we go through another labeling process using these scales as guidance for labeling, or if possible using an automated method (i.e., for commonsense the process above).
	- Now we have: Per-type difficulty number, and we can average this difficulty to have an overall-difficulty for a specific problem.
	- Using our labeled data
		- we can graph relation of per-type difficulty versus accuracy for particular model size classes
		- we can also do this for the overall difficulty
		- if the labeled data is enough, we can experiment with training classifiers to identify difficulty labels and values on other datasets. If the classifiers are accurate/promising enough then we can include them in the paper
		- modeling: we could potentially identify a mathematical model for what models (of a particular size) are not capable of (percent chance of being able to get a question correct -- given K attempts? greedy decoding? self-consistency? etc.)



==Note==: we are only considering difficult problems, so only problems with a certain level of difficulty might get labels for a certain type. But that does not mean other problems don't have this same difficulty (with a lower level) which the problem didn't make errors on and thus wasn't labeled.
- Might be solved if labeling same questions over more models. But you can always consider an even smaller model that might have problems with a difficulty level that is not labeled for a problem because *all* the other models solve it correctly. 


## This week
- Made annotation streamlit ui functional
- Created dataset filtered by the difficulty metric we defined last week
	- Top 100 problems from sorted difficulty
- Thought through experiment + follow-up experiment, definitions for difficulty, how to approach difficulty type labeling
- Annotated 50/100 of the filtered dataset w/ difficulty type + notes on the reasoning for the annotations.
- "Other" category
	- ==Model misunderstands== something clearly stated in the question. What kind of difficulty is this? Is it because the problem is hard to understand?
	- In what way?
		- I don't know how to answer this with a good difficulty type label.
		- question complexity? it what way is it complex? is this very specific based on the wording of the question, or is there a way to generalize this type of difficulty into something that able to be applied to a *wide range of problems*.
- What labels would be needed to make the data significant?
	- It seems difficult to label these. If I come up with a hyper-specific label for a certain question (because the difficulty comes from some specific misunderstanding), are they useful? Does it scale to more problems (or a synthetic solution)?
- It seems making good labels and having data that is labeled well and overall meaningful is very important.