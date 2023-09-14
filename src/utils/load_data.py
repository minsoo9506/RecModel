import numpy as np
import pandas as pd


class EmbeddingIndexGenerater:
    def __init__(
        self, data: pd.DataFrame, user_col: str = "user", item_col: str = "item"
    ):
        """make user&item embedding vector indexer

        Args:
            data (pd.DataFrame): dataframe with columns user&item id
            user_col (str, optional): name for user column. Defaults to "user".
            item_col (str, optional): name for item column. Defaults to "item".
        """
        self.unique_users = data[user_col].unique()
        self.num_unique_users = len(self.unique_users)
        self.unique_items = data[item_col].unique()
        self.num_unique_items = len(self.unique_items)

        self.user_to_idx = {
            original: idx for idx, original in enumerate(self.unique_users)
        }
        self.item_to_idx = {
            original: idx for idx, original in enumerate(self.unique_items)
        }


class RandomNegativeSampler:
    def __init__(self, data: pd.DataFrame, neg_samples_per_pos: int):
        """random negative sampler, random sample in whole non interaction data

        Args:
            data (pd.DataFrame): (user, item, interaction) format dataframe
            neg_samples_per_pos (int): num of negative samples per positive samples
        """
        c = ""
        for col in data.columns:
            c += col
        assert (
            c == "useriteminteraction"
        ), "columns' name should be ['user', 'item', 'interaction']"
        self.data = data
        self.n_users = len(set(data["user"]))
        self.n_items = len(set(data["item"]))
        self.n_pos_samples = len(data)
        self.n_neg_samples = self.n_pos_samples * neg_samples_per_pos
        # non interaction을 모두 모아서 random sample하는 시나리오
        # 따라서, interaction이 적은 user들은 더 많은 negative sample이 뽑힌다.
        self.n_neg_per_user = round(
            (self.n_items - data["user"].value_counts())
            / (self.n_users * self.n_items - self.n_pos_samples)
            * self.n_neg_samples
        ).map(int)
        self.unique_item = set(data["item"])

    def negative_sampling(self, seed: int) -> np.ndarray:
        """negative sampling

        Args:
            seed (int): numpy seed for np.random.choice

        Returns:
            np.ndarray: df added negative samples, (user, item, interaction)
        """
        np.random.seed(seed)
        final_n_neg_samples = 0
        user_neg_samples_dict = {}
        for id, other_features in self.data.groupby("user"):
            pos_samples = other_features["item"].values
            neg_sample_candidates = list(self.unique_item - set(pos_samples))
            neg_item_ids = np.random.choice(
                neg_sample_candidates,
                min(self.n_neg_per_user[id], len(neg_sample_candidates)),
                replace=False,
            )
            user_neg_samples_dict[id] = neg_item_ids
            final_n_neg_samples += len(neg_item_ids)

        df_neg_samples = np.zeros((final_n_neg_samples, 3))

        idx = 0
        for user_item in user_neg_samples_dict.items():
            user = user_item[0]
            for item in user_item[1]:
                df_neg_samples[idx, 0] = user
                df_neg_samples[idx, 1] = item
                idx += 1

        df_pos_samples = np.array(self.data)
        df = np.concatenate((df_pos_samples, df_neg_samples))
        np.random.shuffle(df)
        df = df.astype(np.int64)
        return df
