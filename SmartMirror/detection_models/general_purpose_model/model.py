import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.utils import plot_model


def ResidualTCNBlock(x, filters, kernel_size, dilation_rate, dropout_rate=0.0):
    """
    Functional implementation of a single TCN residual block:
      - Conv1D (causal) → BatchNorm → ReLU → Dropout
      - Conv1D (causal) → BatchNorm → Dropout
      - 1×1 Conv on residual path if channel dims differ
      - Add & ReLU
    """
    # First conv + BN + ReLU + Dropout
    y = layers.Conv1D(filters, kernel_size,
                      dilation_rate=dilation_rate,
                      padding='causal')(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Dropout(dropout_rate)(y)

    # Second conv + BN + Dropout
    y = layers.Conv1D(filters, kernel_size,
                      dilation_rate=dilation_rate,
                      padding='causal')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(dropout_rate)(y)

    # Residual connection (1x1 conv if needed)
    if x.shape[-1] != filters:
        x = layers.Conv1D(filters, 1, padding='same')(x)

    out = layers.add([x, y])
    out = layers.Activation('relu')(out)
    return out


def build_exercise_segmentation_model(
    input_dim,
    num_filters=32,
    kernel_size=3,
    num_layers=4,
    dropout_rate=0.1,
    learning_rate=1e-4
):
    """
    Builds and compiles a TCN-based model for segmentation:
      - Stacks `num_layers` ResidualTCNBlock with exponentially increasing dilation
      - Final 1×1 conv + sigmoid for per-frame probability
      - Uses BinaryFocalCrossentropy for rare-event focus
    """
    inputs = layers.Input(shape=(None, input_dim))
    x = inputs
    for i in range(num_layers):
        dilation = 2 ** i
        x = ResidualTCNBlock(
            x,
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            dropout_rate=dropout_rate
        )

    # Final conv to single-channel output + sigmoid
    x = layers.Conv1D(1, 1, padding='same')(x)
    outputs = layers.Activation('sigmoid')(x)

    model = models.Model(inputs, outputs, name='tcn_segmenter')

    # Focal loss for class imbalance
    loss = BinaryFocalCrossentropy(from_logits=False,
                                   alpha=0.25,
                                   gamma=2.0)
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            metrics.AUC(name='auc'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall')
        ]
    )
    return model


if __name__ == '__main__':
    model = build_exercise_segmentation_model(input_dim=25),
    model.summary()
    plot_model(model, to_file='tcn_segmenter.png', show_shapes=True, expand_nested=True)
