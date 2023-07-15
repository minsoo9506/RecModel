from typing import List

import torch
import torch.nn as nn


class NeuralMatrixFactorization(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_emb_dim: int,
        mlp_emb_dim: int,
        mlp_hidden_dims_list: List[int],
    ):
        """_summary_

        Parameters
        ----------
        num_users : int
            num of unique users
        num_items : int
            num of unique items
        gmf_emb_dim : int
            embedding dimensions of GMF
        mlp_emb_dim : int
            embedding dimensions of MLP
        mlp_hidden_dims_list : List[int]
            list of MLP hidden dimensions
        """
        super().__init__()
        self.gmf_user_emb = nn.Embedding(num_users, gmf_emb_dim)
        self.gmf_item_emb = nn.Embedding(num_items, gmf_emb_dim)
        self.mlp_user_emb = nn.Embedding(num_users, mlp_emb_dim)
        self.mlp_item_emb = nn.Embedding(num_items, mlp_emb_dim)

        mlp = [nn.Linear(2 * mlp_emb_dim, mlp_hidden_dims_list[0]), nn.ReLU()]
        for i in range(1, len(mlp_hidden_dims_list) - 1):
            mlp.append(nn.Linear(mlp_hidden_dims_list[i], mlp_hidden_dims_list[i + 1]))
            mlp.append(nn.ReLU())
        mlp.append(
            nn.Linear(mlp_hidden_dims_list[-2], mlp_hidden_dims_list[-1])
        )  # 마지막에는 RELU 없이
        self.mlp = nn.Sequential(*mlp)

        self.NeuMF_layer = nn.Linear(gmf_emb_dim + mlp_hidden_dims_list[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_

        Parameters
        ----------
        X : torch.Tensor
            user, item input data

        Returns
        -------
        torch.Tensor
            predicted rate (0 ~ 1)
        """
        users, items = X[:, 0], X[:, 1]

        element_wise_prod = torch.mul(
            self.gmf_user_emb(users), self.gmf_item_emb(items)
        )
        mlp_concat = torch.cat(
            (self.mlp_user_emb(users), self.mlp_item_emb(items)), dim=1
        )
        mlp_out = self.mlp(mlp_concat)
        final_layer_input = torch.cat((element_wise_prod, mlp_out), dim=1)
        out = self.NeuMF_layer(final_layer_input).reshape(-1)
        return out
