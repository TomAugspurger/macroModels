class StepFun:

    def __init__(self, X, Y):
        """
        """
        self.X, self.Y = X, Y

    def __call__(self, x):
        """
        """
        if x < self.X[0]:
            return 0.0
        i = self.X.searchsorted(x, side='right') - 1
        return self.Y[i]

    def expectation(self, F):
        """
        """
        probs = np.append(F(self.X), 1)
        return np.dot(self.Y, probs[1:] - probs[:-1])
