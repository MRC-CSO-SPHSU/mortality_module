import numpy as np


def kullback_leibler_divergence(pd : np.ndarray,
                                reference_pd : np.ndarray) -> float:
    # size check of some sort
    return np.multiply(pd, np.log(np.divide(pd,
                                            reference_pd,
                                            out=np.ones_like(pd),
                                            where=reference_pd!=0))).sum()


