import numpy as np
import scipy


def log_loss(pred, y):
    return np.log(np.exp(-pred * y) + 1.0)


class FactorizationMachines:
    def __init__(self, X: scipy.sparse._csr.csr_matrix, y: np.ndarray, config: dict):
        """FM init

        Args:
            X (scipy.sparse._csr.csr_matrix): input data
            y (np.ndarray): input label
            config (dict): config
        """
        # data: 값
        # indices: '0'이 아닌 원소의 열 위치
        # indptr : 행 위치 시작
        # https://rfriend.tistory.com/551
        # implicit이라는 MF 라이브러리에서도 csr사용

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

    def _predict_1_instance(
        self, i: int, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """predict 1 data instance

        Args:
            i (int): data index
            data (np.ndarray): csr_matrix.data
            indices (np.ndarray): csr_matrix.indices
            indptr (np.ndarray): csr_matrix.indptr

        Returns:
            tuple[float, np.ndarray]: prediction result, summed result
        """

        summed = np.zeros(self.n_factors)
        summed_squared = np.zeros(self.n_factors)

        # bias
        pred = self.w0

        # linear: w * x
        for idx in range(indptr[i], indptr[i + 1]):
            feature_col_loc = indices[idx]
            pred += self.W[feature_col_loc] * data[idx]

        # interaction
        for factor in range(self.n_factors):
            for idx in range(indptr[i], indptr[i + 1]):
                feature_col_loc = indices[idx]
                # row를 먼저 for문으로 도는게 비효율적일 수 있으나 일단 논문과 동일한 형태의 V를 만들기 위해
                term = self.V[feature_col_loc, factor] * data[idx]
                summed[factor] += term
                summed_squared[factor] += term**2

            pred += 0.5 * (summed[factor] ** 2 - summed_squared[factor])

        return pred, summed

    def _sgd(self) -> float:
        """sgd update

        Returns:
            float: averaged 1 epoch train loss
        """
        loss = 0.0

        for i in range(self.n_samples):
            pred, summed = self._predict_1_instance(
                i, self.data, self.indices, self.indptr
            )

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

    def fit(self) -> list[float]:
        """train model

        Returns:
            list[float]: list of epoch loss
        """
        epoch_loss = []
        for epoch in range(self.epochs):
            loss = self._sgd()
            print(f"[epoch: {epoch+1}], loss: {loss}")
            epoch_loss.append(loss)

        return epoch_loss

    def predict(self, X: scipy.sparse._csr.csr_matrix) -> np.ndarray:
        """predict data ratings

        Args:
            X (scipy.sparse._csr.csr_matrix): input data to predict ratings

        Returns:
            np.ndarray: pred result
        """
        data = X.data
        indices = X.indices
        indptr = X.indptr
        pred_result = X.shape[0]

        # bias
        pred = self.w0

        for i in range(X.shape[0]):
            pred, _ = self._predict_1_instance(i, data, indices, indptr)
            pred_result[i] = pred

        return pred_result
