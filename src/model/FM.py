from typing import List, Tuple

import numpy as np
import scipy


def log_loss(pred, y):
    return np.log(np.exp(-pred * y) + 1.0)


class FactorizationMachine:
    def __init__(self, X: scipy.sparse._csr.csr_matrix, y: np.ndarray, config: dict):
        """init FM

        Parameters
        ----------
        X : scipy.sparse._csr.csr_matrix
            input data
        y : np.ndarray
            input label
        config : dict
            _description_
        """
        # data: 값
        # indices: '0'이 아닌 원소의 열 위치
        # indptr : 행 위치 시작
        # https://rfriend.tistory.com/551

        self.data = X.data
        self.indices = X.indices
        self.indptr = X.indptr
        self.y = y

        self.epochs = config["epochs"]
        self.n_factors = config["n_factors"]
        self.learning_rate = config["learning_rate"]
        self.regularize_W = config["regularize_W"]
        self.regularize_V = config["regularize_V"]

        self.n_samples, self.n_features = X.shape
        self.w0 = 0.0  # bias
        self.W = np.random.normal(size=self.n_features)  # weight
        self.V = np.random.normal(size=(self.n_features, self.n_factors))

    def _predict(self, i: int) -> Tuple[float, np.ndarray]:
        """predict rating

        Parameters
        ----------
        i : int
            index of data

        Returns
        -------
        Tuple[float, np.ndarray]
            pred, summed
        """

        summed = np.zeros(self.n_factors)
        summed_squared = np.zeros(self.n_factors)

        # bias
        pred = self.w0

        # linear: w * x
        for idx in range(self.indptr[i], self.indptr[i + 1]):
            feature_col_loc = self.indices[idx]
            pred += self.W[feature_col_loc] * self.data[idx]

        # interaction
        for factor in range(self.n_factors):
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                feature_col_loc = self.indices[idx]
                # row를 먼저 for문으로 도는게 비효율적일 수 있으나 일단 논문과 동일한 형태의 V를 만들기 위해
                term = self.V[feature_col_loc, factor] * self.data[idx]
                summed[factor] += term
                summed_squared[factor] += term**2

            pred += 0.5 * (summed[factor] ** 2 - summed_squared[factor])

        return pred, summed

    def _sgd(self) -> float:
        """sgd update

        Returns
        -------
        float
            averaged loss in 1 epoch
        """
        loss = 0.0

        for i in range(self.n_samples):
            pred, summed = self._predict(i)

            # calculate loss
            loss += log_loss(pred, self.y[i])
            loss_grad = -self.y[i] / (np.exp(self.y[i] * pred) + 1.0)

            # update bias
            self.w0 -= self.learning_rate * loss_grad

            # update W
            for idx in range(self.indptr[i], self.indptr[i + 1]):
                feature_col_loc = self.indices[idx]
                self.W[feature_col_loc] -= self.learning_rate * (
                    loss_grad * self.data[idx]
                    + 2 * self.regularize_W * self.W[feature_col_loc]
                )

            # update V
            for factor in range(self.n_factors):
                for idx in range(self.indptr[i], self.indptr[i + 1]):
                    feature_col_loc = self.indices[idx]
                    term = (
                        summed[factor]
                        - self.V[feature_col_loc, factor] * self.data[idx]
                    )
                    V_grad = loss_grad * self.data[idx] * term
                    self.V[feature_col_loc, factor] -= self.learning_rate * (
                        V_grad + 2 * self.regularize_V * self.V[feature_col_loc, factor]
                    )

        loss /= self.n_samples

        return loss

    def fit(self) -> List[float]:
        epoch_loss = []
        for epoch in range(self.epochs):
            loss = self._sgd()
            print(f"[epoch: {epoch+1}], loss: {loss}")
            epoch_loss.append(loss)

        return epoch_loss

    def predict(self, X: scipy.sparse._csr.csr_matrix) -> np.ndarray:
        """predict

        Parameters
        ----------
        X : scipy.sparse._csr.csr_matrix
            input data to predict ratings

        Returns
        -------
        np.ndarray
            prediction result
        """

        data = X.data
        indices = X.indices
        indptr = X.indptr

        summed = np.zeros(self.n_factors)
        summed_squared = np.zeros(self.n_factors)

        pred_result = np.zeros(data.shape[0])

        # bias
        pred = self.w0

        for i in range(data.shape[0]):

            # linear: w * x
            for idx in range(indptr[i], indptr[i + 1]):
                feature_col_loc = indices[idx]
                pred += self.W[feature_col_loc] * data[idx]

            # interaction
            for factor in range(self.n_factors):
                for idx in range(indptr[i], indptr[i + 1]):
                    feature_col_loc = indices[idx]
                    term = self.V[feature_col_loc, factor] * data[idx]
                    summed[factor] += term
                    summed_squared[factor] += term**2

                pred += 0.5 * (summed[factor] ** 2 - summed_squared[factor])

        pred_result[i] = pred

        return pred_result
