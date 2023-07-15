# reference: https://github.com/albertauyeung/matrix-factorization-in-python

import numpy as np


class MF:
    def __init__(
        self, matrix: np.ndarray, latent_dim: int, alpha: float, beta: float, iters: int
    ) -> None:
        """matrix factorization with SGD

        Parameters
        ----------
        matrix : np.ndarray
            user-item matrix
        latent_dim : int
            latent dimension
        alpha : float
            learning rate
        beta : float
            regularization parameter
        iters : int
            num of iterations
        """
        self.matrix = matrix
        self.num_users, self.num_items = matrix.shape
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta
        self.iters = iters

    def train(self) -> None:
        """train MF"""

        # initialize user and item latent feature matrix
        self.U = np.random.normal(
            scale=1.0 / self.latent_dim, size=(self.num_users, self.latent_dim)
        )
        self.I = np.random.normal(
            scale=1.0 / self.latent_dim, size=(self.num_items, self.latent_dim)
        )

        # initialize the bias
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.matrix[np.where(self.matrix != 0)])

        # create a list of train sample: not zero
        self.samples = [
            (i, j, self.matrix[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.matrix[i, j] != 0
        ]

        # SGD
        for i in range(self.iters):
            np.random.shuffle(self.samples)
            self._optim_sgd()
            rmse = self._get_rmse()
            print(f"Iteration: {i + 1}; train_rmse = {rmse:.5f}")

    def _get_rmse(self) -> float:
        """get rmse

        Returns
        -------
        float
            rmse
        """
        xs, ys = self.matrix.nonzero()
        predicted = self._get_current_matrix()
        mse = np.mean(np.power(self.matrix[xs, ys] - predicted[xs, ys], 2))
        rmse = np.sqrt(mse)
        return rmse

    def _get_current_matrix(self) -> np.ndarray:
        """get current updating(predict) matrix

        Returns
        -------
        np.ndarray
            current predict matrix
        """
        return (
            self.b
            + self.b_u[:, np.newaxis]
            + self.b_i[
                np.newaxis :,
            ]
            + self.U.dot(self.I.T)
        )

    def _optim_sgd(self) -> None:
        """perform sgd"""
        for i, j, r in self.samples:
            prediction = self._get_rating(i, j)
            e = r - prediction

            # Update biases
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

            # Create copy of row of U since we need to update it but use older values for update on I
            U_i = self.U[i, :][:]

            # Update user and item latent feature matrices
            self.U[i, :] += self.alpha * (e * self.I[j, :] - self.beta * self.U[i, :])
            self.I[j, :] += self.alpha * (e * U_i - self.beta * self.I[j, :])

    def _get_rating(self, i: int, j: int) -> float:
        """get the predicted rating of user i and item j

        Parameters
        ----------
        i : int
            user index
        j : int
            item index

        Returns
        -------
        float
            predicted rating
        """
        prediction = (
            self.b + self.b_u[i] + self.b_i[j] + self.U[i, :].dot(self.I[j, :].T)
        )
        return prediction

    def show_full_matrix(self) -> np.ndarray:
        """show full matrix

        Returns
        -------
        np.ndarray
            result(predicted) matrix
        """
        return (
            self.b
            + self.b_u[:, np.newaxis]
            + self.b_i[
                np.newaxis :,
            ]
            + self.U.dot(self.I.T)
        )
