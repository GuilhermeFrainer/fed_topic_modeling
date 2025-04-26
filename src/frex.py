import numpy as np


def exclusivity(topics: np.ndarray, top_words=10, exclusivity_weight=0.7) -> float:
    """
    Computes FREX score for each topic.

    Parameters
    ----------
    topics : np.ndarray
        Topic matrix where each row is a topic and each column is a word within a topic.
        Values represent probability that word is in topic.
        Returned by `model.get_topics()` in gensim.
    
    top_words : int, optional
        Number of top words to consider per topic. Default `10`.

    exclusivity_weight : float, optional
        Weight parameter towards exclusivity. Default `0.7`.

    Notes
    -----
    This function is essentially a port of an R function from the stm package.
    Source code available here: https://github.com/bstewart/stm/blob/master/R/exclusivity.R.
    """
    # Small error to avoid division by zero
    e = 1e-10
    
    # Normalize across columns
    # Document this part a bit better. Why am I doing this?
    col_sums = topics.sum(axis=0)
    exclusivity_matrix = topics / col_sums # col_sums[np.newaxis, :] might be more general

    num_topics = topics.shape[0]

    ex = __ecdf(exclusivity_matrix)
    fr = __ecdf(topics)

    frex = 1.0 / (exclusivity_weight / (ex + e) + (1 - exclusivity_weight) / (fr + e))
    # Gets indices for words with highest frequencies per topic
    top_indices = np.argsort(-topics, axis=1)[:, :top_words]
    scores = [frex[i, top_indices[i]].sum() for i in range(num_topics)]
    return scores


def __ecdf(arr: np.ndarray) -> np.ndarray:
    r"""
    Empirical Cumulative Distribution Function.

    Parameters
    ----------
    arr : np.ndarray
        Array for which you want to compute the ECDF.

    Returns
    -------
    np.ndarray
        ECDF of the array.

    Examples
    --------
    Example generated (but not written by) ChatGPT.
    >>> values = [0.1, 0.2, 0.05, 0.65]
    >>> ranks = rank(values)
    >>> ranks
    [2, 3, 1, 4]

    >>> ecdf = ranks / len(ranks)
    >>> ecdf
    [0.5, 0.75, 0.25, 1.0]

    That is, there is a 0.25 probability that a variable x will take on a value
    less than or equal to 0.05, as seen by the fact that the value 0.25 is in the
    same position in the ecdf vector as 0.05 was in the values vector.
    This is true, since there's 1 value less than or equal to 0.05 in the original
    vector out of 4 values. $1 / 4 = 0.25$.

    Similarly, the value 0.2 is larger than or equal to 3 of the values.
    Thus ECDF(0.2) = 0.75.

    Notes
    -----
    A Cumulative Distribution Function (CDF) is the probability that a random
    variable X will take on a value less than or equal to x. That is

    $F_X(x) = P(X \leq x)$.

    Since we are not working with a continuous function but with actual, discrete data,
    we can do something simpler:

    $ECDF(x) = \frac{#{x_i \leq x}}{n}$

    that is, the probability that a variable x will take on a value less than or equal to x
    is the number of values less than or equal to x divided by all the values in a given vector.
    """
    n = arr.shape[1]
    return __rank(arr) / n


def __rank(arr: np.ndarray) -> np.ndarray:
    """
    Returns the sample ranks of the values in a vector.
    Replicates R's rank function with default options.
    See Notes for explanation.

    Parameters
    ----------
    arr : np.ndarray
        Array to be ranked. Will be ranked along axis=0.
    
    Returns
    -------
    np.ndarray
        Rank vector

    Notes
    -----
    The following R code
    ```r
    x <- c(20, 30, 10, 4, 65)
    rank(x)
    ```
    returns
    `[1] 3 4 2 1 5`.

    So now `ranked_x[i]` gives us the position that the element `x[i]` should have to sort the array.
    I.e., "the i-th element would be in the ranked_x[i]-th position in a sorted vector".

    Conversely, the following Python code
    ```python
    x = np.array([20, 30, 10, 4, 65])
    np.argsort(x) + 1 # Adding one to match the R output, which is 1-indexed
    ```
    returns
    `[4 3 1 2 5]`

    Which is not the same thing. In this case, `ranked_x[i]` gives us the index of the element
    that would have to occupy `i` in a sorted array. I.e., "if I want to create a sorted array,
    I have to go back to `x`, pick its i-th element, and put it in this position".

    Running `np.argsort` once more seems to fix this. I'm not sure why. See:
    ```python
    x = np.array([20, 30, 10, 4, 65])
    sorted_x = np.argsort(x)
    np.argsort(sorted_x) + 1 # Adding one to match the R output, which is 1-indexed
    ```
    This outputs:
    `[3 4 2 1 5]`
    which is equal to the R output.
    """
    return np.argsort(np.argsort(arr, axis=0), axis=0)

