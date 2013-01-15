class Stochastic:
    def __init__(self, z, T):
        """
        z is an array of shocks.
        T is the transition matrix.
        """
        self.z = z
        self.T = T
        self.size = len(z)
