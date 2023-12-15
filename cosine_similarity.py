from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
import numpy as np  # type: ignore


def calculate_cosine_similarity(str_list_1: list, str_list_2=None):
    """
    Return a two-dimensional array representing the cosine similarity scores between the documents in str_list_1 and str_list_2.

    If str_list_2 is None or empty, the function calculates the similarity between str_list_1 and itself.

    Parameters:
    - str_list_1 (list): A list of strings representing the first set of documents.
    - str_list_2 (list, optional): A list of strings representing the second set of documents. If not provided or empty, the function calculates the similarity between str_list_1 and itself.

    Returns:
    - numpy.ndarray: A two-dimensional array of similarity scores.

    Example Usage:
    ```
    sim = calculate_cosine_similarity(['White House', '', 'S&P'], str_list_2=['White House', 'Donald Trump', 'S P', 'S&P'])
    print(sim)
    ```
    Output:
    ```
    [[1.         0.         0.        ]
     [0.         1.         0.        ]
     [0.         0.         1.        ]]
    ```
    """
    if not str_list_1:
        return np.array([[0.0]])
    if not str_list_2:
        vectors = get_vectors(str_list_1)
        return cosine_similarity(vectors[: len(str_list_1)])
    else:
        vectors = get_vectors(str_list_1 + str_list_2)
        len_x = len(str_list_1)
        return cosine_similarity(
            vectors[:len_x], Y=vectors[len_x : len(str_list_1 + str_list_2)]
        )


def get_vectors(str_list: list, hash_threshold: int = 500):
    """
    Returns a matrix of vectors representing the text in the input strings.

    Args:
        str_list (list): A list of strings representing the input text.
        hash_threshold (int, optional): The threshold value for using HashingVectorizer. Defaults to 500.

    Returns:
        numpy.ndarray: A matrix of vectors representing the text in the input strings.
    """

    text = [t.replace("&", "_") for t in str_list]
    if len(text) > hash_threshold:
        vectorizer = HashingVectorizer(n_features=10000, stop_words=None)
        try:
            m = vectorizer.fit_transform(text)
        except ValueError:
            text += ["random_string_a_p_w"]
            m = vectorizer.fit_transform(text)
    else:
        vectorizer = CountVectorizer(stop_words=None)
        try:
            vectorizer.fit(text)
        except ValueError:
            text += ["random_string_a_p_w"]
            vectorizer.fit(text)
        m = vectorizer.transform(text)
    return m.toarray()


if __name__ == "__main__":
    import time

    start = time.time()
    sim = calculate_cosine_similarity(
        ["White House", "", "S&P"],
        str_list_2=["White House", "Donald Trump", "S P", "S&P"],
    )
    end = time.time()
    print("time {}".format(end - start))
    print(sim)
