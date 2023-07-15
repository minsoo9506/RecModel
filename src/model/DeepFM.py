from typing import List

import numpy as np
import torch
import torch.nn as nn

# reference: torchFM
# 여기서 구현한 모델은 0,1로만으로 field가 이루어져있는 경우로 가정하고 진행


class FMLinear(nn.Module):
    def __init__(self, field_dims: List[int], output_dim: int = 1):
        """linear part in FM component

        Parameters
        ----------
        field_dims : List[int]
            dimension of each field
        output_dim : int, optional
            always 1 because it is for linear term, by default 1
        """
        super().__init__()

        self.fc = nn.Embedding(sum(field_dims), output_dim)
        self.bias = nn.parameter(torch.zeros((output_dim,)))
        self.offsets = np.array(
            (0, *np.cumsum(field_dims)[:-1]), dtype=np.int_
        )  # 새로운 종류의 field가 시작하는 index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        torch.Tensor
            value of linear term
        """
        # |x| = (batch_size, num_fields)
        # 여기서 num_fields는 각 종류의 field안에서 user, item의 각 index
        # 그래서 offset을 더해줘야 embedding layer에서 원하는 weight를 뽑아낼 수 있다
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeatureEmbedding(nn.Module):
    def __init__(self, field_dims: List[int], embed_dim: int):
        """embedding part for FM and Deep Component

        Parameters
        ----------
        field_dims : List[int]
            dimension of each field
        embed_dim : int
            embedding dimensions
        """
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int_)
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Parameters
        ----------
        x : torch.Tensor
            input data

        Returns
        -------
        torch.Tensor
            x's embedding vectors
        """
        # |x| = (batch_size, num_fields)
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FMInteraction(nn.Module):
    def __init__(self):
        """interaction term in FM"""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # |x| = (batch_size, num_fields, embed_dim)
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x**2, dim=1)
        ix = square_of_sum - sum_of_square
        ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dims: List[int],
        dropout: float,
        output_layer: bool = True,
    ):
        """MLP part in Deep Component

        Parameters
        ----------
        input_dim : int
            _description_
        embed_dims : List[int]
            _description_
        dropout : float
            _description_
        output_layer : bool, optional
            _description_, by default True
        """
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Parameters
        ----------
        x : torch.Tensor
            input data (embedding vectors)

        Returns
        -------
        torch.Tensor
            _description_
        """
        # |x| = (batch_size, embed_dim)
        return self.mlp(x)


class DeepFM(nn.Module):
    def __init__(
        self, field_dims: List[int], embed_dim: int, mlp_dims: List[int], dropout: float
    ):
        """DeepFM model

        Parameters
        ----------
        field_dims : List[int]
            dimension of each field
        embed_dim : int
            embedding dimensions
        mlp_dims : List[int]
            _description_
        dropout : float
            _description_
        """
        super().__init__()
        self.fm_linear = FMLinear(field_dims)
        self.fm_interaction = FMInteraction()
        self.embedding = FeatureEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MLP(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Parameters
        ----------
        x : torch.Tensor
            _description_

        Returns
        -------
        torch.Tensor
            predicted ratings
        """
        # |x| = (batch_size, num_fields)
        embed_x = self.embedding(x)
        x = (
            self.fm_linear(x)
            + self.fm_interaction(embed_x)
            + self.mlp(embed_x.view(-1, self.embed_output_dim))
        )
        out = torch.sigmoid(x.squeeze(1))
        # |out| = (batch_size, )
        return out
