import torch
import torch.nn as nn

# google paper와는 조금 다르게 embedding을 share하는 형태가 아님
# user, item two-tower 시나리오 가정


class TowerModel(nn.Module):
    def __init__(
        self,
        embedding_vocab_size: dict[str, int],
        embedding_dim: int,
        num_continuous_feature: int,
        layer_dims: list[tuple],
    ) -> None:
        """Tower class of Two-Tower model

        Args:
            embedding_vocab_size (dict[str, int]): vocab size of each embedding layer, {'name': vocab_size}
            embedding_dim (int): dimension of embedding layer
            num_continuous_feature (int): num of continuous feature
            layer_dims (list[tuple]): dim of nn.Linear layer
        """
        super().__init__()
        assert (
            len(embedding_vocab_size) * embedding_dim + num_continuous_feature
            == layer_dims[0][0]
        ), "first input dim of feed-foward network is wrong"

        self.num_continuous_feature = num_continuous_feature

        self.embeddings = []
        for num_embeddings in list(embedding_vocab_size.values()):
            self.embeddings.append(nn.Embedding(num_embeddings, embedding_dim))

        layers = []
        for in_features, out_features in layer_dims[:-1]:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(torch.nn.ReLU())
        layers.append(nn.Linear(layer_dims[-1][0], layer_dims[-1][1]))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x_cate: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            x_cate (torch.Tensor): categoric feature
            x_num (torch.Tensor): numetic feature

        Returns:
            torch.Tensor: vector before inner product
        """
        concat_embeddings = torch.cat(
            [
                embedding(x_cate[:, idx])
                for idx, embedding in enumerate(self.embeddings)
            ],
            dim=1,
        )
        input = torch.cat([concat_embeddings, x_num], dim=1)
        output = self.layers(input)
        output = nn.functional.normalize(output, p=2, dim=1)
        return output


class TwoTower(nn.Module):
    def __init__(self, query_model: TowerModel, candidate_model: TowerModel) -> None:
        """Two-Tower model

        Args:
            query_model (TowerModel): query TowerModel
            candidate_model (TowerModel): candidate TowerModel
        """
        super().__init__()
        self.query_model = query_model
        self.candidate_model = candidate_model

    def forward(
        self,
        x: tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """forward

        Args:
            x (torch.Tensor): input (query, candidate)

        Returns:
            torch.Tensor: prediction
        """
        query, candidate = x
        query_model_output = self.query_model(query[0], query[1])
        candidate_model_output = self.candidate_model(candidate[0], candidate[1])
        output = torch.sum(query_model_output * candidate_model_output, dim=1)
        return output
