"""
Segmentation Model Module

Defines the Temporal Convolutional Network (TCN) architecture for exercise repetition detection.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.utils import plot_model


def ResidualTCNBlock(x, filters, kernel_size, dilation_rate, dropout_rate=0.0):
    """
    Functional implementation of a TCN residual block.
    
    Architecture:
      - Conv1D (causal) → BatchNorm → ReLU → Dropout
      - Conv1D (causal) → BatchNorm → Dropout
      - 1×1 Conv on residual path if channel dims differ
      - Add & ReLU
    
    Args:
        x: Input tensor
        filters: Number of output filters
        kernel_size: Size of the convolution kernel
        dilation_rate: Dilation rate for the convolution
        dropout_rate: Dropout rate (default: 0.0)
    
    Returns:
        Output tensor after residual block
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

    # Residual connection (1×1 conv if needed)
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
    Builds and compiles a TCN-based model for exercise repetition segmentation.
    
    The model predicts per-frame probabilities of exercise repetitions using:
    - Stacked residual TCN blocks with exponential dilation
    - Final 1×1 conv + sigmoid for per-frame probability output
    - Binary Focal Crossentropy loss for class imbalance handling
    
    Args:
        input_dim: Number of input features per frame (typically 25 joint angles)
        num_filters: Number of filters in TCN layers
        kernel_size: Size of the convolution kernel
        num_layers: Number of TCN layers
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model for exercise segmentation
    """
    # Input layer
    inputs = layers.Input(shape=(None, input_dim))
    x = inputs
    
    # Stack TCN layers with exponential dilation
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

    # Create model
    model = models.Model(inputs, outputs, name='tcn_segmenter')

    # Compile with focal loss for class imbalance
    loss = BinaryFocalCrossentropy(
        from_logits=False,
        alpha=0.25,
        gamma=2.0
    )
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


def get_model_summary(input_dim=25):
    """
    Get a summary of the model architecture.
    
    Args:
        input_dim: Number of input features
        
    Returns:
        Model summary string
    """
    model = build_exercise_segmentation_model(input_dim=input_dim)
    return model.summary()


def save_model_plot(input_dim=25, output_path='tcn_segmenter.png'):
    """
    Save a visual plot of the model architecture.
    
    Args:
        input_dim: Number of input features
        output_path: Path to save the plot
    """
    model = build_exercise_segmentation_model(input_dim=input_dim)
    plot_model(
        model, 
        to_file=output_path, 
        show_shapes=True, 
        expand_nested=True
    )
    print(f"Model architecture plot saved to: {output_path}")


if __name__ == '__main__':
    """Example usage and model visualization."""
    print("Building exercise segmentation model...")
    
    # Create model
    model = build_exercise_segmentation_model(input_dim=25)
    
    # Print summary
    print("\nModel Summary:")
    model.summary()
    
    # Save architecture plot
    save_model_plot(input_dim=25, output_path='tcn_segmenter.png')
    
    print("\nModel created successfully!") 