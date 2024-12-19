import nltk
import spacy
from typing import List, Set

nlp = spacy.load("en_core_web_sm")


def detect_concepts(sentence: str, concepts: List[str]) -> Set[str]:
    present_concepts = []

    # Tokenize the sentence and lemmatize the tokens
    tokens = nltk.word_tokenize(sentence)
    lemmas = [token.lemma_ for token in nlp(sentence)]

    # Check if each concept is present in the sentence
    for concept in concepts:
        if concept in tokens or concept in lemmas:
            present_concepts.append(concept)

    return set(present_concepts)


def eval_common_gen(concepts: List[str], response: str):
    present_concepts = detect_concepts(response, concepts)
    return len(present_concepts) / len(concepts)
