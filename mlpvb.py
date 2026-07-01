
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import Input, layers
from tensorflow.keras.layers import (
    Layer, Dense, Dropout, Flatten, Concatenate, Add, Reshape,
    LayerNormalization, MultiHeadAttention, Embedding, Conv2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2, VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
self_learning_Rate_Value = 2
self_learning_Rate_Value2 = 4

ADAMLR = tf.keras.optimizers.Adam(
    learning_rate=(5e-4) / self_learning_Rate_Value + (1e-4) / self_learning_Rate_Value2
)

epochs = 3
learning_rate = 0.001
decay_rate = learning_rate / epochs

image_size1 = 256
train_dir = "/content/datasetval/train"
test_dir = "/content/datasetval/test"


# ---------------------------------------------------------------------------
# Model building blocks
# ---------------------------------------------------------------------------
class ClassToken(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, 1, input_shape[-1]), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]
        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls


def mlp(x, cf):
    x = Dense(cf["mlp_dim"], activation="relu")(x)
    x = Dropout(cf["dropout_rate"])(x)
    x = Dense(cf["hidden_dim"])(x)
    x = Dropout(cf["dropout_rate"])(x)
    return x


def transformer_encoder(x, cf):
    skip_1 = x
    x = LayerNormalization()(x)
    x = MultiHeadAttention(num_heads=cf["num_heads"], key_dim=cf["hidden_dim"])(x, x)
    x = Add()([x, skip_1])

    skip_2 = x
    x = LayerNormalization()(x)
    x = mlp(x, cf)
    x = Add()([x, skip_2])
    return x


def build_vgg_branch(image_size):
    """VGG19 feature-extraction branch (frozen weights)."""
    vgg = VGG19(include_top=False, weights="imagenet",
                input_shape=(image_size, image_size, 3))
    for layer in vgg.layers:
        layer.trainable = False  # was `layers.trainable = False` (no-op) in the original

    inputs = Input((image_size, image_size, 3))
    vgg_out = Flatten()(vgg(inputs))
    x = Dense(32, activation=tf.keras.activations.gelu)(vgg_out)
    x = Dense(64, activation=tf.keras.activations.gelu)(x)
    x = Dense(128, activation=tf.keras.activations.gelu)(x)
    x = Dense(256, activation=tf.keras.activations.gelu)(x)
    x = Dense(256, activation=tf.keras.activations.gelu)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return Model(inputs, x, name="vgg19_branch")


def MobileNetV2ViT(cf, vgg_branch):
    """
    Hybrid model: MobileNetV2 patch embedding -> ViT encoder stack,
    fused with a VGG19 feature branch, as described in the paper
    abstract.
    """
    inputs = Input((cf["image_size"], cf["image_size"], cf["num_channels"]))

    # --- MobileNetV2 + ViT branch ---
    basemodel = MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs)
    output = basemodel.output

    patch_embed = Conv2D(cf["hidden_dim"], kernel_size=cf["patch_size"], padding="same")(output)
    _, h, w, f = patch_embed.shape
    patch_embed = Reshape((h * w, f))(patch_embed)

    positions = tf.range(start=0, limit=h * w, delta=1)
    pos_embed = Embedding(input_dim=h * w, output_dim=cf["hidden_dim"])(positions)
    embed = patch_embed + pos_embed
    token = ClassToken()(embed)

    x = Concatenate(axis=1)([token, embed])
    for _ in range(cf["num_layers"]):
        x = transformer_encoder(x, cf)

    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    vit_out = x[:, 0, :]  # CLS token representation

    # --- VGG19 branch, fed the *same* input tensor ---
    vgg_features = vgg_branch(inputs)

    # --- Fuse both branches (this was previously discarded) ---
    fused = Concatenate(axis=-1)([vit_out, vgg_features])
    fused = Dense(1, activation="sigmoid")(fused)

    model = Model(inputs, fused, name="MLPVB")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = {
        "mode": "train",  # or "test"  (was accidentally set to the train_dir path)
        "num_layers": 18,
        "hidden_dim": 60,
        "mlp_dim": 60,
        "num_heads": 12,
        "dropout_rate": 0.1,
        "image_size": image_size1,
        "patch_size": 32,
        "num_channels": 3,
        "num_classes": 2,
    }
    config["num_patches"] = (config["image_size"] ** 2) // (config["patch_size"] ** 2)

    vgg_branch = build_vgg_branch(image_size1)
    model = MobileNetV2ViT(config, vgg_branch)
    model.summary()

    model.compile(optimizer=ADAMLR, loss="binary_crossentropy", metrics=["accuracy"])

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode="nearest",
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size1, image_size1),
        batch_size=32,
        class_mode="binary",
    )
    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(image_size1, image_size1),
        batch_size=32,
        class_mode="binary",
        shuffle=False,
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
    )
