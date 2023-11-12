import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
from keras.src.engine.keras_tensor import KerasTensor


class RepresentLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        int_vocabs: dict[str, np.ndarray],
        str_vocabs: dict[str, np.ndarray],
        statistics: dict[str, tuple[float, float]],
    ):
        super().__init__()

        self.int_cate_cols = int_vocabs.keys()
        self.str_cate_cols = str_vocabs.keys()
        self.conti_cols = statistics.keys()

        self.int_representation_layer = self.create_int_representation_layer(int_vocabs)
        self.str_representation_layer = self.create_str_representation_layer(str_vocabs)
        self.conti_representation_layer = self.create_conti_representation_layer(
            statistics
        )

    def create_int_representation_layer(
        self, int_vocabs: dict[str, np.ndarray]
    ) -> tf.keras.Model:
        inputs = [
            tf.keras.Input(shape=(), name=col, dtype=tf.int64)
            for col in int_vocabs.keys()
        ]
        outputs = []
        for x in inputs:
            vocab = int_vocabs[x.name]
            lookup = tf.keras.layers.IntegerLookup(vocabulary=vocab)(x)
            embedding = tf.keras.layers.Embedding(vocab.size + 1, 8)(lookup)
            outputs.append(embedding)
        concat = tf.keras.layers.Concatenate()(outputs)
        return tf.keras.Model(inputs=inputs, outputs=concat)

    def create_str_representation_layer(
        self, str_vocabs: dict[str, np.ndarray]
    ) -> tf.keras.Model:
        inputs = [
            tf.keras.Input(shape=(), name=col, dtype=tf.string)
            for col in str_vocabs.keys()
        ]
        outputs = []
        for x in inputs:
            vocab = str_vocabs[x.name]
            lookup = tf.keras.layers.StringLookup(vocabulary=vocab)(x)
            embedding = tf.keras.layers.Embedding(vocab.size + 1, 8)(lookup)
            outputs.append(embedding)
        concat = tf.keras.layers.Concatenate()(outputs)
        return tf.keras.Model(inputs=inputs, outputs=concat)

    def create_conti_representation_layer(
        self, statistics: dict[str, tuple[float, float]]
    ) -> tf.keras.Model:
        inputs = [
            tf.keras.Input(shape=(), name=col, dtype=tf.float32)
            for col in statistics.keys()
        ]
        outputs = []
        for x in inputs:
            mean, var = statistics[x.name]
            normalized = tf.keras.layers.Normalization(
                axis=-1, mean=mean, variance=var
            )(tf.expand_dims(x, axis=1))
            outputs.append(normalized)
        concat = tf.keras.layers.Concatenate()(outputs)
        return tf.keras.Model(inputs=inputs, outputs=concat)

    def call(self, inputs: tf.keras.Input) -> KerasTensor:
        x_str_cate = self.str_representation_layer(
            {key: inputs[key] for key in self.str_cate_cols}
        )
        x_int_cate = self.int_representation_layer(
            {key: inputs[key] for key in self.int_cate_cols}
        )
        x_conti = self.conti_representation_layer(
            {key: inputs[key] for key in self.conti_cols}
        )

        return tf.concat([x_str_cate, x_int_cate, x_conti], axis=-1)


class TowerModel(tf.keras.Model):
    def __init__(
        self,
        layer_size_list: list[int],
        int_vocabs: dict[str, np.ndarray],
        str_vocabs: dict[str, np.ndarray],
        statistics: dict[str, tuple[float, float]],
    ):
        super().__init__()
        self.embedding_layer = RepresentLayer(int_vocabs, str_vocabs, statistics)

        self.dense_layer = tf.keras.Sequential()
        for layer_size in layer_size_list[:-1]:
            self.dense_layer.add(tf.keras.layers.Dense(layer_size, activation="relu"))
        self.dense_layer.add(tf.keras.layers.Dense(layer_size_list[-1]))

    def call(self, inputs):
        feature_embedding = self.embedding_layer(inputs)
        out = self.dense_layer(feature_embedding)
        return out


class TwoTower(tfrs.models.Model):
    def __init__(
        self,
        user_int_vocabs: dict[str, np.ndarray],
        user_str_vocabs: dict[str, np.ndarray],
        user_statistics: dict[str, tuple[float, float]],
        user_features: list[str],
        user_dense_layer_size_list: list[int],
        item_int_vocabs: dict[str, np.ndarray],
        item_str_vocabs: dict[str, np.ndarray],
        item_statistics: dict[str, tuple[float, float]],
        item_features: list[str],
        item_dense_layer_size_list: list[int],
        tf_dataset_item,
    ):
        super().__init__()
        self.query_model = TowerModel(
            user_dense_layer_size_list,
            user_int_vocabs,
            user_str_vocabs,
            user_statistics,
        )
        self.candidate_model = TowerModel(
            item_dense_layer_size_list,
            item_int_vocabs,
            item_str_vocabs,
            item_statistics,
        )
        self.user_features = user_features
        self.item_features = item_features

        self.task = tfrs.tasks.Retrieval(
            loss=tf.keras.losses.CategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
            ),
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=tf_dataset_item.batch(128).map(self.candidate_model),
            ),
        )

    def compute_loss(self, features, training=False):
        query_embedding = self.query_model(
            {key: features[key] for key in features.keys() if key in self.user_features}
        )

        candidate_embedding = self.candidate_model(
            {key: features[key] for key in features.keys() if key in self.item_features}
        )

        out = self.task(
            query_embedding, candidate_embedding, compute_metrics=not training
        )
        return out
