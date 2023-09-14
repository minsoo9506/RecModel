import numpy as np
import torch
import torch.nn as nn

# 여기서 구현한 모델은 0,1로만으로 field가 이루어져있는 경우로 가정하고 진행
# 따라서 continuous field도 이산화 후 원핫인코딩 필요
# field: user, item과 같은 여러개의 feature가 만들어진 것을 의미


class FMLinear(nn.Module):
    def __init__(self, field_dims: list[int]):
        """linear part in FM component

        Args:
            field_dims (list[int]): dimension of each field
        """
        super().__init__()

        self.fc = nn.Embedding(sum(field_dims), 1)
        self.offsets = np.array(
            (0, *np.cumsum(field_dims)[:-1]), dtype=np.int_
        )  # 새로운 종류의 field가 시작하는 index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): input data, |x| = (batch_size, num_field)

        Returns:
            torch.Tensor: result, |result| = (batch, 1)
        """

        # 여기서 instance input은 각 종류의 field안에서 해당 값의 index
        # 그래서 offset을 더해줘야 embedding layer에서 원하는 weight를 뽑아낼 수 있다
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1)


class FeatureEmbedding(nn.Module):
    def __init__(self, field_dims: list[int], embed_dim: int):
        """embedding part for FM and Deep Component

        Args:
            field_dims (list[int]): dimension of each field
            embed_dim (int): dimension of embedding
        """
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int_)
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): input data, |x| = (batch_size, num_field)

        Returns:
            torch.Tensor: embeddings of a instance input, |output| = (batch_size, len(field_dims), embed_dim)
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FMInteraction(nn.Module):
    def __init__(self):
        """interaction term in FM"""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): embeddings, |x| = (batch_size, num_field, embed_dim)

        Returns:
            torch.Tensor: interaction term result in FM, |output| = (batch_size, 1)
        """
        # 아래 연산은 FM 논문에서 증명
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x**2, dim=1)
        ix = square_of_sum - sum_of_square
        ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class Deep(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_layers: int, dropout: float):
        """Deep component in DeepFM

        Args:
            input_dim (int): len(field_dims) * embed_dim
            output_dim (int): output dimesion
            n_layers (int): num of feed-forward layer
            dropout (float): dropout rate
        """
        super().__init__()
        layers = list()
        for _ in range(n_layers):
            layers.append(torch.nn.Linear(input_dim, output_dim))
            layers.append(torch.nn.BatchNorm1d(output_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = output_dim
        layers.append(torch.nn.Linear(output_dim, 1))
        self.deep = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): embedding layer output, |x| = (batch_size, len(field_dims) * embed_dim)

        Returns:
            torch.Tensor: |output| = (batch_size, 1)
        """
        return self.deep(x)


class DeepFM(nn.Module):
    def __init__(
        self,
        field_dims: list[int],
        embed_dim: int,
        deep_output_dim: int,
        deep_n_layers: int = 1,
        deep_dropout: float = 0.2,
    ):
        """DeepFM

        Args:
            field_dims (list[int]): dimension of each field
            embed_dim (int): embedding dimensions
            deep_output_dim (int): output dimension of feed-forward layer in Deep component
            deep_n_layers (int, Optional): num of feed-forward layers in Deep component. Defaults to 1.
            deep_dropout (float, Optional): dropout rate in Deep component. Defaults to 0.2.
        """
        super().__init__()
        self.fm_linear = FMLinear(field_dims)
        self.fm_interaction = FMInteraction()
        self.embedding = FeatureEmbedding(field_dims, embed_dim)
        self.Deep_input_dim = len(field_dims) * embed_dim
        self.deep = Deep(
            self.Deep_input_dim, deep_output_dim, deep_n_layers, deep_dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): input data, |x| = (batch_size, num_field)

        Returns:
            torch.Tensor: final CTR prediction. |out| = (batch_size, )
        """
        embed_x = self.embedding(x)
        # |embed_x| = (batch_size, len(field_dims), embed_dim)
        x = (
            self.fm_linear(x)
            + self.fm_interaction(embed_x)
            + self.deep(embed_x.view(-1, self.Deep_input_dim))
        )
        out = torch.sigmoid(x.squeeze(1))
        return out
