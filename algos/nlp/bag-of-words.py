from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import jax
import jax.numpy as jnp
import nltk
from loguru import logger
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

K = TypeVar("K")


@jax.jit
def softmax(arr: jnp.ndarray) -> jnp.ndarray:
    """Apply softmax function to a numpy array"""
    return jnp.exp(arr - jnp.max(arr)) / jnp.sum(jnp.exp(arr - jnp.max(arr)))


class SuffixTree:
    @dataclass(slots=True)
    class Node:
        count: int = 0
        children: dict = None

        def __post_init__(self):
            if self.children is None:
                self.children = {}

    def __init__(self):
        self.root = {}

    def add_sentence(self, sentence: Sequence[str]) -> None:
        """Add all suffixes of a sentence to the tree"""
        for i in range(len(sentence)):
            suffix = sentence[i:]
            current = self.root
            for j, word in enumerate(suffix):
                if word not in current:
                    current[word] = self.Node()
                if j == len(suffix) - 1:
                    current[word].count += 1
                else:
                    current = current[word].children

    def completions(self, sequence: Sequence[str]) -> Mapping[str, int]:
        """Get counts for all possible completions of a sequence"""
        current = self.root
        for word in sequence:
            if word not in current:
                return {}
            current = current[word].children
        return {word: node.count for word, node in current.items()}

    def predict_completion(
        self,
        sequence: Sequence[str],
        key: jax.random.PRNGKey,
        vocab: Sequence[str],
        k: float = 1.0,
        boost_factor: float = 2.0,
    ) -> tuple[str, jax.random.PRNGKey]:
        """Predict next word using exponential boosting of existing sequences"""
        completions = self.completions(sequence)
        key, next_key = jax.random.split(key)
        if not completions:
            return vocab[jax.random.choice(key, jnp.arange(len(vocab)))], next_key

        probabilities = softmax(
            jnp.array(
                [(completions.get(word, 0) + k) ** boost_factor for word in vocab]
            )
        )

        chosen_idx = int(jax.random.categorical(key, logits=jnp.log(probabilities)))
        return vocab[chosen_idx], next_key

    def add_sentences(self, sentences: Collection[Sequence[str]]) -> None:
        """Add all sentences to the suffix tree"""
        for sentence in sentences:
            self.add_sentence(sentence)


def main() -> None:
    nltk.download("stopwords")
    nltk.download("punkt_tab")

    corpus = Path("data/shakespeare.txt").read_text()

    raw_sentences = sent_tokenize(corpus)
    logger.info(f"Number of sentences: {len(raw_sentences)}")

    stop_words = set(stopwords.words("english"))

    sentences = [
        cleaned_tokens
        for tokens in (word_tokenize(sent) for sent in raw_sentences)
        if (
            cleaned_tokens := [
                cleaned
                for word in tokens
                if word.lower() not in stop_words
                and (cleaned := word.strip("\ufeff\x00-\x1f\x7f-\x9f"))
                and cleaned.isprintable()
            ]
        )
    ]

    vocab = sorted(set(word for sentence in sentences for word in sentence))
    logger.info(f"Number of unique words: {len(vocab)}")

    model = SuffixTree()
    model.add_sentences(sentences)

    key = jax.random.PRNGKey(0)

    completion_sequence = ["A", "rose", "by", "any", "other"]
    completions = model.completions(completion_sequence)
    logger.info(f"Completions for {completion_sequence}: {completions}")
    for i in range(10):
        predicted, key = model.predict_completion(
            completion_sequence, key, vocab, boost_factor=100
        )
        completion_sequence.append(predicted)
        logger.info(f"Predicted completion for {completion_sequence}: {predicted}")


if __name__ == "__main__":
    main()
