from IncFedMDRS.utils.calc_P_online import calc_P_online
from IncFedMDRS.utils.subsample import subsample
import numpy as np
from copy import deepcopy
from IncFedMDRS.layers import Input, Reservoir
from numpy import linalg as LA
from sklearn.decomposition import PCA


class MDRS:
    def __init__(
        self,
        N_u,
        N_x,
        input_scale,
        rho,
        leaking_rate,
        delta,
        trans_len,
        precision_matrix=None,
        N_x_tilde=None,
        threshold=None,
        density=0.05,
        activation_func=np.tanh,
        noise_level=None,
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
        self.trans_len = trans_len
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

        if N_x_tilde is None:
            N_x_tilde = N_x

        self.N_x_tilde = N_x_tilde

        if precision_matrix is None:
            self.precision_matrix = (1.0 / self.delta) * np.eye(N_x_tilde, N_x_tilde)
        else:
            self.precision_matrix = precision_matrix

    def train(self, U):
        """
        U: input data
        """
        covariance_matrix = self.delta * np.identity(self.N_x_tilde)
        train_length = len(U)

        for n in range(train_length):
            x_in = self.Input(U[n])

            if self.noise is not None:
                x_in += self.noise

            x = self.Reservoir(x_in)

            if n > self.trans_len:
                x = x.reshape((-1, 1))
                x = subsample(x, self.N_x_tilde, self.seed)

                covariance_matrix += np.dot(x, x.T)

                # disable comment out below when you perform online learning
                # self.precision_matrix = calc_P_online(self.precision_matrix, x)
                #
                # mahalanobis_distance = np.dot(np.dot(x.T, self.precision_matrix), x)
                # self.threshold = (
                #     max(mahalanobis_distance, self.threshold)
                #     if self.threshold is not None
                #     else mahalanobis_distance
                # )

        return covariance_matrix

    def train_with_PCA(self, U, n_components=1):
        covariance_matrix = self.delta * np.identity(self.N_x_tilde)
        train_length = len(U)

        for n in range(train_length):
            x_in = self.Input(U[n])

            if self.noise is not None:
                x_in += self.noise

            x = self.Reservoir(x_in)

            if n > self.trans_len:
                x = x.reshape((-1, 1))
                x = subsample(x, self.N_x_tilde, self.seed)

                covariance_matrix += np.dot(x, x.T)

                # disable comment out below when you perform online learning
                # self.precision_matrix = calc_P_online(self.precision_matrix, x)
                #
                # mahalanobis_distance = np.dot(np.dot(x.T, self.precision_matrix), x)
                # self.threshold = (
                #     max(mahalanobis_distance, self.threshold)
                #     if self.threshold is not None
                #     else mahalanobis_distance
                # )

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        return eigenvalues[:n_components], eigenvectors[:, :n_components]

    def adapt(self, U, threshold=None):
        """
        U: input data
        """
        data_length = len(U)
        mahalanobis_distances = []

        if threshold is not None:
            self.threshold = threshold

        for n in range(data_length):
            x_in = self.Input(U[n])

            x = self.Reservoir(x_in)
            x = x.reshape((-1, 1))
            x = subsample(x, self.N_x_tilde, self.seed)

            mahalanobis_distance = np.dot(np.dot(x.T, self.precision_matrix), x)
            mahalanobis_distance = np.squeeze(mahalanobis_distance)
            mahalanobis_distances.append(mahalanobis_distance)

        return np.array(mahalanobis_distances, dtype=np.float64)

    def set_P(self, P):
        self.precision_matrix = P

    def copy(self):
        return deepcopy(self)
