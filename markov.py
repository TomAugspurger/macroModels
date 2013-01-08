import numpy as np


def markov(T, n=100, s0=0, V=None):
    """Need to check if V should be index zero or one.
    Move to a class.  Also read up on numba
    """
    r, c = np.shape(T)
    if r != c:
        raise Exception

    if V == None:
        V = np.array(range(r))

    if T.sum(axis=1).all() != 1:
        raise Exception

    if np.shape(V) != (r,):
        raise Exception

    x = np.random.rand(n)
    s = np.zeros([r, 1])
    s[s0] = 1
    cum = np.dot(T, np.triu(np.ones(T.shape)))
    state = np.zeros([r, len(x)])
    for i, v in enumerate(x):
        state[:, i] = s.T
        ppi = np.concatenate([np.array([[0.]]), np.dot(s.T, cum)], axis=1)
        s = ((v <= ppi[0, 1:r + 1]) * (v > ppi[0, 0:r])).reshape(r, 1).astype(float)

    return (np.dot(V, state), state)
