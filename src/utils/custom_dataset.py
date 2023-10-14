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


class TwoTowerDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        user_num_cols: list[str],
        item_num_cols: list[str],
        user_cate_cols: list[str],
        item_cate_cols: list[str],
        label: str,
    ):
        """init

        Args:
            data (pd.DataFrame): data include feature and label
            user_num_cols (list[str]): numeric user columns' name
            item_num_cols (list[str]): numeric item columns' name
            user_cate_cols (list[str]): categoric user columns' name
            item_cate_cols (list[str]): categoric item columns' name
            label (str): label(target)'s name
        """

        user_cols = user_cate_cols + user_num_cols
        item_cols = item_cate_cols + item_num_cols

        self.user = data[user_cols]
        self.item = data[item_cols]

        for user_cate_col in user_cate_cols:
            user_to_idx = {
                original: idx
                for idx, original in enumerate(self.user[user_cate_col].unique())
            }
            self.user[user_cate_col] = self.user[user_cate_col].map(user_to_idx)

        for item_cate_col in item_cate_cols:
            item_to_idx = {
                original: idx
                for idx, original in enumerate(self.item[item_cate_col].unique())
            }
            self.item[item_cate_col] = self.item[item_cate_col].map(item_to_idx)

        self.user = torch.from_numpy(self.user.values).type(torch.float32)
        self.item = torch.from_numpy(self.item.values).type(torch.float32)
        self.y = torch.from_numpy(data[label].values).type(torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user[idx, :], self.item[idx, :], self.y[idx]
