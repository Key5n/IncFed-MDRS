import numpy as np
import networkx as nx
from copy import deepcopy


def identity(x):
    return x


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


class MDRS:
    def __init__(
        self,
        N_u,
        N_x,
        threshold=None,
        density=0.05,
        input_scale=1.0,
        rho=0.95,
        activation_func=np.tanh,
        leaking_rate=1.0,
        noise_level=None,
        delta=0.0001,
        update=1,
        lam=1,
        seed=0,
    ):
        self.seed = seed
        self.Input = Input(N_u, N_x, input_scale, seed=self.seed)
        self.Reservoir = Reservoir(
            N_x, density, rho, activation_func, leaking_rate, seed=self.seed
        )
        self.N_u = N_u
        self.N_x = N_x
        self.threshold = None if threshold == None else threshold
        self.precision_matrix = None
        if noise_level is None:
            self.noise = None
        else:
            np.random.seed(seed=0)
            self.noise = np.random.uniform(-noise_level, noise_level, (self.N_x, 1))
        self.delta = delta
        self.lam = lam
        self.update = update
        self.precision_matrix = (1.0 / self.delta) * np.eye(N_x, N_x)

    def train(self, U, trans_len=0):
        """
        U: input data
        """
        local_updates = np.zeros((self.N_x, self.N_x), dtype=np.float64)
        train_length = len(U)

        for n in range(train_length):
            x_in = self.Input(U[n])

            if self.noise is not None:
                x_in += self.noise

            x = self.Reservoir(x_in)

            if n > trans_len:
                x = x.reshape((-1, 1))
                self.precision_matrix = self.calc_next_precision_matrix(
                    x, self.precision_matrix
                )
                local_updates += np.dot(x, x.T)

                mahalanobis_distance = np.dot(np.dot(x.T, self.precision_matrix), x)
                self.threshold = (
                    max(mahalanobis_distance, self.threshold)
                    if self.threshold is not None
                    else mahalanobis_distance
                )

        return local_updates

    def adapt(self, U, threshold=None):
        """
        U: input data
        """
        data_length = len(U)
        label = []
        mahalanobis_distances = []

        if threshold is not None:
            self.threshold = threshold

        for n in range(data_length):
            x_in = self.Input(U[n])

            x = self.Reservoir(x_in)
            mahalanobis_distance = np.dot(np.dot(x.T, self.precision_matrix), x)
            # print(f"{mahalanobis_distance = }")
            mahalanobis_distances.append(mahalanobis_distance)

            if mahalanobis_distance < self.threshold:
                self.precision_matrix = self.calc_next_precision_matrix(
                    x, self.precision_matrix
                )
                label.append(0)
            else:
                # mark the data as anomalous
                label.append(1)

        return np.array(label, dtype=np.int8), np.array(
            mahalanobis_distances, dtype=np.float64
        )

    def calc_next_precision_matrix(self, x, precision_matrix):
        x = np.reshape(x, (-1, 1))
        next_precision_matrix = precision_matrix
        for _ in np.arange(self.update):
            gain = 1 / self.lam * np.dot(next_precision_matrix, x)
            gain = gain / (
                1 + 1 / self.lam * np.dot(np.dot(x.T, next_precision_matrix), x)
            )
            next_precision_matrix = (
                1
                / self.lam
                * (
                    next_precision_matrix
                    - np.dot(np.dot(gain, x.T), next_precision_matrix)
                )
            )
        return next_precision_matrix

    def set_P(self, P):
        self.precision_matrix = P

    def copy(self):
        return deepcopy(self)
