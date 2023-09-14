import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NeuMFDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # torch.Tensor()는 새 메모리를 할당하지만 torch.from_numpy()는 그대로 사용한다고 함
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).type(torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx, :], self.y[idx]


class DeepFMDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        """dataset for DeepFM model

        Parameters
        ----------
        data : pd.DataFrame
            _description_
        """
        super().__init__()

        user_to_index = {
            original: idx for idx, original in enumerate(data.user.unique())
        }
        movie_to_index = {
            original: idx for idx, original in enumerate(data.movie.unique())
        }
        data["user"] = data["user"].apply(lambda x: user_to_index[x])
        data["movie"] = data["movie"].apply(lambda x: movie_to_index[x])
        # [user, movie, rate] -> (user, movie, rate)
        data = data.to_numpy()[:, :3]

        self.items = data[:, :2].astype(np.intc)
        self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
        self.field_dims = np.max(self.items, axis=0) + 1

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        return self.items[idx], self.targets[idx]

    def __preprocess_target(self, target):
        target[target <= 9] = 0
        target[target > 9] = 1
        return target
