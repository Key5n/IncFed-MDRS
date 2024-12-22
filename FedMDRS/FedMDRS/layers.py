import numpy as np
import networkx as nx
from copy import deepcopy


class Input:
    def __init__(self, N_u, N_x, input_scale, seed=0):
        """
        param N_u: input dim
        param N_x: reservoir size
        param input_scale: input scaling
        """
        # uniform distribution
        np.random.seed(seed=seed)
        self.Win = np.random.uniform(-input_scale, input_scale, (N_x, N_u))

    # weighted sum
    def __call__(self, u):
        """
        param u: (N_u)-dim vector
        return: (N_x)-dim vector
        """
        return np.dot(self.Win, u)

    def copy(self):
        return deepcopy(self)


class Reservoir:
    def __init__(self, N_x, density, rho, activation_func, leaking_rate, seed=0):
        """
        param N_x: reservoir size
        param density: connection density
        param rho: spectral radius
        param activation_func: activation function
        param leaking_rate: leak rates
        param seed
        """
        self.seed = seed
        self.W = self.make_connection(N_x, density, rho)
        self.x = np.zeros(N_x)
        self.activation_func = activation_func
        self.alpha = leaking_rate

    def make_connection(self, N_x, density, rho):
        # Erdos-Renyi random graph
        m = int(N_x * (N_x - 1) * density / 2)
        G = nx.gnm_random_graph(N_x, m, self.seed)
        connection = nx.to_numpy_array(G)
        W = np.array(connection)

        rec_scale = 1.0
        np.random.seed(seed=self.seed)
        W *= np.random.uniform(-rec_scale, rec_scale, (N_x, N_x))

        # rescaling
        eigv_list = np.linalg.eig(W)[0]
        sp_radius = np.max(np.abs(eigv_list))
        W *= rho / sp_radius

        return W

    def __call__(self, x_in):
        """
        param x_in: x before update
        return: x after update
        """
        self.x = np.multiply(1.0 - self.alpha, self.x) + np.multiply(
            self.alpha, self.activation_func(np.dot(self.W, self.x) + x_in)
        )
        return self.x

    def copy(self):
        return deepcopy(self)
