import torch
import torch.nn as nn


class NeuralMatrixFactorization(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_emb_dim: int,
        mlp_emb_dim: int,
        mlp_hidden_dim_list: list[int],
    ):
        """NeuMF model (no share embedding vectore between GMF and MLP)

        Args:
            num_users (int): num of users
            num_items (int): num of items
            gmf_emb_dim (int): dimension of GMF embedding vector
            mlp_emb_dim (int): dimension of MLP embedding vector
            mlp_hidden_dim_list (list[int]): list of hidden layer of MLPs' dimension
        """
        super().__init__()
        assert (
            mlp_emb_dim * 2 == mlp_hidden_dim_list[0]
        ), "should be: mlp_emb_dim * 2 = mlp_hidden_dim_list[0]"
        self.gmf_user_emb = nn.Embedding(num_users, gmf_emb_dim)
        self.gmf_item_emb = nn.Embedding(num_items, gmf_emb_dim)
        self.mlp_user_emb = nn.Embedding(num_users, mlp_emb_dim)
        self.mlp_item_emb = nn.Embedding(num_items, mlp_emb_dim)

        mlp = []
        for i in range(0, len(mlp_hidden_dim_list) - 2):
            mlp.append(nn.Linear(mlp_hidden_dim_list[i], mlp_hidden_dim_list[i + 1]))
            mlp.append(nn.ReLU())
        mlp.append(
            nn.Linear(mlp_hidden_dim_list[-2], mlp_hidden_dim_list[-1])
        )  # 마지막에는 RELU 없이
        self.mlp = nn.Sequential(*mlp)

        self.NeuMF_layer = nn.Linear(gmf_emb_dim + mlp_hidden_dim_list[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): |x| = (batch_size, 2), index of user & item embedding vector

        Returns:
            torch.Tensor: |out| = (batch,)
        """
        users, items = x[:, 0], x[:, 1]

        element_wise_prod = torch.mul(
            self.gmf_user_emb(users), self.gmf_item_emb(items)
        )
        mlp_concat = torch.cat(
            (self.mlp_user_emb(users), self.mlp_item_emb(items)), dim=1
        )
        mlp_out = self.mlp(mlp_concat)
        final_layer_input = torch.cat((element_wise_prod, mlp_out), dim=1)
        out = self.NeuMF_layer(final_layer_input).reshape(-1)
        out = self.sigmoid(out)
        return out
