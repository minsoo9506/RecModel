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
            embedding = tf.keras.layers.Embedding(vocab.size, 8)(lookup)
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
            embedding = tf.keras.layers.Embedding(vocab.size, 8)(lookup)
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
    pass


class TwoTower(tf.keras.Model):
    pass
