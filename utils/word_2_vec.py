from typing import List, Tuple, Dict

import numpy as np
from scipy import spatial
import pprint
import os
import pickle
from tqdm import tqdm


def load_data(file_path: str) -> List[str]:
    words = []
    with open(file_path, "r") as file:
        line = file.readline()
        while line:
            line = line.lower()
            line = line.replace("\n", " ")
            line = ''.join(e for e in line if e.isalnum() or e == " ")
            line_words = line.split(" ")
            line_words = [word for word in line_words if word != "" and len(word) < 20]
            words.extend(line_words)
            line = file.readline()
    return words


def count_word(word: str, corpus: List[str]) -> int:
    return sum([corpus_word == word for corpus_word in corpus])


def sort_word_counts(word_counts: List[Tuple[str, int]]):
    word_counts.sort(key=lambda x: x[1], reverse=True)


def count_coocurences(wordA: str, wordB: str, corpus: List[str], window_size: int = 1) -> int:
    A_indices = list(np.where([w == wordA for w in corpus])[0])
    B_indices = list(np.where([w == wordB for w in corpus])[0])
    A_named_indices = [(index, "A") for index in A_indices]
    B_named_indices = [(index, "B") for index in B_indices]
    named_indices = A_named_indices + B_named_indices
    named_indices.sort(key=lambda x: x[0])
    previous_index = -window_size - 1
    previous_name = ""

    coocurence_count = 0

    for named_index in named_indices:
        index, name = named_index
        if index - previous_index <= window_size and name != previous_name:
            coocurence_count += 1

        previous_index = index
        previous_name = name

    return coocurence_count


def distance(word_vec_A: List[float], word_vec_B: List[float]) -> float:
    return spatial.distance.cosine(word_vec_A, word_vec_B)


def get_words_sorted_by_similarity(word: str, w2v: Dict[str, List[float]]) -> List[str]:
    word_distances = [(w, distance(w2v[w], w2v[word])) for w in w2v.keys()]
    word_distances = [(w, dist) for w, dist in word_distances if not np.isnan(dist)]
    word_distances.sort(key=lambda x: x[1])
    return [w for w, dist in word_distances]


if __name__ == "__main__":
    text_corpus = load_data()
    unique_words = list(set(text_corpus))
    unique_word_counts = [(word, count_word(word, text_corpus)) for word in unique_words]
    sort_word_counts(unique_word_counts)
    print(unique_word_counts)
