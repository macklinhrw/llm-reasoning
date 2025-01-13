---
date: 2025-01-12
tags:
  - unique-note
  - projects/llm-reasoning
aliases:
  - At what point are problems too difficult for LLMs and can we model this?
---
related:: [[2025-01-12]]

# 202501120746
---

## What do we not know?
- We don't know whether a model can get a problem correct or not before it generates an answer.
	- I believe we won't be able to know this unless we understood to some extent the composition and quality of the finetune/pretrain data (which we won't know). Technical papers do tend to list composition of data such as categorization, but we would need a consistent way of profiling datasets (which are already too large) across different models.
- Exactly what is the cutoff point in difficulty? Past a certain level of difficulty when will the model definitely (no matter the number of generations) be unable to get a problem correct?
	- Difficulty is very difficult to define
	- And simplifying it to a degree where I can manage it might not result in a paper that discovers anything more or different than what the reasoning boundary paper did

## Is it worth answering?
- Lets say I was able to answer this question and we now know the level of difficulty (a metric I defined) a model is unlikely to ever to be able to answer correctly.
- First, in what way would I answer it?
	- (statistic analysis?)
	- We can graph threshold values of difficulty for models that are different sizes then overall make some conclusions as to the correlation of size of model to the this threshold value
	- For a particular model, if we run it on an evaluation task with well labeled difficulties that are part of different difficulty categories, we can analyze the different difficulties that are thresholds for the model.
	- How do combinations of difficulties affect the model?
	- After the evaluation is run, we can graph all these relations, and the result will be a profile of the model, the affect difficulty has on its ability to get problems right, and an estimated upper bound of difficulty it will never get problem correct after.

## Difficulty (some exploration)
- Commonsense
	- How "common" is the commonsense?
		- Is this something we can even answer?
- Reasoning steps
	- How many required reasoning steps
		- How do we determine this in a definite way?
- Computational complexity
	- How big are the numbers involved in arithmetic operations

## Compound difficulty
- If we simplify the difficulty measure to just looking at number of reasoning steps (or any singular measure), what if there are correlations of problems with higher number of reasoning steps to being difficult in other ways?
	- This would be a confounding variable and impact any correlations we want to make with just this singular difficulty metric, making them faulty.

## Experiment details
- Determine different types of difficulty categories, and how they contribute to an incorrect reasoning step.
	- ref: [[verify-step-by-step.pdf]]
	- *Iterate over all the reasoning steps until a stop condition*.
	- Did the model make an error at a particular step?
		- Yes, No, Neutral
	- *Iterate through each possible error category*
	- Each has a potential score of 0 through 3 (whether the model failed at this step due to this error category)
		- 0 - No
		- 1 - Maybe
		- 2 - Agree
		- 3 - Strongly Agree
	- For each X with score > 1, rate the difficulty of X in the problem: match the closest bound that is correct.
		- 1 - All humans could get this correct
		- 2 - Almost all humans could get this correct
		- 3 - The average human could get this correct
		- 4 - An undergraduate student could get this correct
		- 5 - A graduate student could get this correct
		- (6 - An expert in the subject could get this correct?)
	- If we had multiple annotators we could take the majority voted difficulty or average them if there is no consensus
	- Why not annotate the difficulty of a problem directly instead of looking at step-wise errors?
		- Looking at real errors gives annotators a concrete basis for difficulty: why did the model get this wrong, what was difficult? As opposed to: what could be difficult?
		- The latter question is more open-ended, I think the former will give better data because the annotators have concrete information to base their annotations off of.

## Oversight?
- This is annotated data based on what the model got wrong. However, there may be difficult parts of the problem that the model got correct, which will end up not being annotated due to this approach.
- In fact, there is some of this inherent in whatever labels I choose, I can't expect them to be perfect and exhaustive of all the types of difficulty that may be encountered. This means that there will possibly difficulty that is unaccounted for and could potentially impact my experiment results.
	- In this respect, I think the experiment is still good because the annotator will answer how strongly they feel the error category contributed to the error. This means with high values for this annotation value we can say that for these problems there is less chance for there to be other difficulty sources contributing to the error. And if we only look at values with agreement (2+) this should also be true. I think some careful consideration is necessary here. Maybe some experiments (I'm not sure what kind yet) could be done to analyze uncertainty or correlation (I believe there are statistical tools, but I would have to explore more to to determine which to use and understand how to apply them here).
- How does this impact my previous ideas?
	- So effectively, with this experiment, we are only looking at errors that contribute to models getting answers (or reasoning steps) incorrect.
	- We can't make concrete conclusions about _all_ the different types of difficulty that went into the problem, but we could make conclusions about what error types could contribute to the model getting the answer incorrect.
	- This slightly changes the meaning or significance if we were to train a model to predict the difficulty types present in a problem. We would instead be training the model to predict difficulty types in a problem that would contribute to the model getting the problem incorrect. Furthermore, there it is likely not to be exhaustive, meaning weaker models that the ones whose answers we are analyzing could have errors in ways that were not present in the annotated generations.
		- As a result, it would be better if the dataset we are annotating (generations GSM8K or MATH) includes generations by many different sizes and classes of models to be able to have a diverse representation of the possible errors all different kinds of models could possibly make.
		- **This means, even if one model doesn't struggle with a category of difficulty, others probably will, and it will be annotated somewhere in the dataset.**

## Another oversight?
- The error categories I previously suggested aren't the same as difficulty categories. The model may make a type of error, but it may not be straight forward how we are able to link that to type of difficulty inherent in the problem.
	- Examples: The model hallucinates, the model has a logical inconsistency, format error (parsing), or computes a variable too early... What type of difficulty in the problem do all these errors originate from?
	- **Issue**: We know the model made an error, but we don't know why.
- What are the difficulty categories?
	- Maybe I can't imagine them all right now, but I should get an idea of whether they exist so I can judge whether the experiment is possible or not.
	- Categories
		- Misunderstood the problem
		- Commonsense
		- Arithmetic
		- Minimum Required Reasoning Steps (Needs concrete examples)
			- How do we know this is the cause of the model error?
			- 
	- I think the categories above have some potential. However, I would need to do some more thinking an analysis to really refine the best possible difficulty categories.
	- Not just solution length, which I hypothesized, but all of these could use some concrete examples.
	- Scaling
		- The issue I'm seeing now is I'm not sure how we would scale these
		- Actually, I think this is accounted for in the experiment, because the difficulty annotation directions are not in relation to the model, they are in relation to _humans_.
		- Different size models will have more issues with solution length, commonsense, arithmetic, etc.
			- Their ability to solve more difficult problems in these categories goes up. How much? What is their limit?

## Based on this experimental data
- So now, what do we have?
	- Annotations
		- Which step did the model make a mistake in the reasoning steps
		- Due to what kind of difficulty category did the model make (with a score of 0-3 on degree of belief)
		- How difficult was the problem with respect to this error category?
			- thought: hallucination is not a type of difficulty inherent to a problem, instead it is a type of error the model may make. It might be difficult, in a problem where a model hallucinates, to judge what kind of difficulty in the problem caused the model to hallucinate.
			- We could add an "Other" error category
		- Then we could take the average of the difficulties for each error category and call this the average or overall difficulty label (a single number).
	- After creating annotations for many different model generations of different sizes and families, we can get an idea of where the difficulty in the problems actually evaluated comes from.
		- The overall difficulty number, coming from a large and diverse set of annotations, should give us a general idea of the difficulty of a problem, and in what ways it is difficult.
- We could evaluate models on the original problems paired with the difficulty annotations and look at correlations between difficulty (overall, individual) and the model's ability to get a question correct.
- We could take this annotated data and train a model to predict the difficulty value and types of difficulty a problem will have.
- This data will also give us step-wise accuracy labels. So we will know, based on the evaluated model(s), the step-wise accuracy of the model.
	- Is it worth doing some type of comparison in looking at end-accuracy versus step-wise accuracy and their relation to difficulty?
	- Question: Will training a step-wise error predictor with the extra feature of error types be more successful?

## The ideal experiment
- Ideally we would have multiple annotators and a large dataset of problems to annotate.
- We would want a dataset with a diverse level of difficulties as well as difficulties that are at least somewhat evenly distributed (some easy problems, some difficult ones).
- Datasets
	- The `train` set of GSM8K has ~7.5k problems
		- I don't know how much the difficulty varies and what is the upper bound of difficult
		- I would hypothesize that the difficulty of problems will not be too much more difficult than grade school math questions based on the name
			- If there is an associated research paper, reading it would help
	- The dataset MATH, used in [[verify-step-by-step.pdf]], has "12,500 challenging competition mathematics problems" ([from the abstract](https://arxiv.org/pdf/2103.03874)).
	- In an ideal world, since these two datasets are encompassing lower-level and higher-level difficulties, annotating both could strengthen any arguments we make about correlations between difficulty and model accuracy (either end or step-wise).
	- I didn't have any particular reason for choosing MATH other than that it seems difficult (more-so than GSM8K), and I have seen it listed in benchmark tables for models before.
		- I could explore other evaluation datasets.
- ...

## A realistic experiment
- Annotators are expensive, and annotating 20k problems is not something I can feasibly do alone.
- A potential way of annotating data is taking a much larger "expert" model, one that achieves a very high (close to 100%) accuracy on the evaluation dataset.
- We could prompt it in a pipeline to synthetically annotate the data.
- If the synthetic approach isn't promising, then there are various ways of simplifying the annotations and reducing the time needed
	- Less error categories
	- Less problems (taking a subset of problems from GSM8K and MATH)
	- If this still isn't enough, the rating scale could be simplified and rethought
- What is realistic for me?
	- I think annotating more than a couple hundred (~200) starts to become unrealistic in a short amount of time-- it will also devour my research time that could be spent in other ways.

## What is the scope?
- Due to the limitations for annotating a dataset (or cost of the expert model described above), the scale of the annotation dataset would have to be smaller, meaning it might not be as conclusive compared to if we were to have a large number of human annotators (and budget).
- Specifically the models I want to look at are smaller ones (where we know the exact size) that will produce meaningful error data.
- A range of models should be used for generating the pre-annotated data.
	- If I am doing manual annotation: I think just looking at the llama series is more manageable.
	- If we have a method to scale more: I think looking at different families of models would be better for more diverse errors. (Such as Qwen-series.)
- If I'm going to train some sort of model based on the data, I should only focus on training one model. This will help simplify the process and not spread myself too thin.
	- I thought of two different types of predictors before -- one for difficulty in a problem, one for step-wise accuracy. I think focusing _just_ on predicting difficulty and error categories is more significant and reasonable.
- Two different potential applications: 
	- we have (step-wise) annotations for model generations
		- We can now to some extent analyze this model
		- How would we use this step-wise accuracy?
		- Instead could I simplify the whole experiment, only look at the end-answer, and change it to only annotate base on end-answer instead of specific-step errors.
		- How would I 
	- we have difficulty values for problems which we can then evaluate other models