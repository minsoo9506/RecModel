import numpy as np


def make_binary(x: np.ndarray) -> np.ndarray:
    """make ratings to binary

    Parameters
    ----------
    x : np.ndarray
        rating matrix

    Returns
    -------
    np.ndarray
        rating matrix with only 0, 1
    """
    return np.where(x != 0, 1, 0)
