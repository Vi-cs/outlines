from typing import List, Sequence

import numpy as np
import scipy.spatial as spatial


def cosine_similarity(
    vectors: Sequence[np.ndarray], query: np.ndarray, k: int = 1
) -> List[np.ndarray]:
    """Use cosine similarity to retrieve the `top_n` closest vectors to the query.

    Be mindful that Scipy computes the cosine distance, defined as one minus the cosine
    similarity.

    Parameters
    ----------
    vectors
        A sequence that contains the vectors to search from.
    query
        The vector whose nearest neighbour we want to find.
    k
        The number of closest matches to return.

    """
    #print('#### BEGIN cosine_similarity')
    #print('Input : ')
    #print(Sequence[np.ndarray])
    #print(np.ndarray)
    similarities = [spatial.distance.cosine(v, query) for v in vectors]
    top_n_indices = np.argsort(similarities)[:k]
    ##print('Ouput : ')
    #print(top_n_indices)
    #print('#### END cosine_similarity')
    return top_n_indices
