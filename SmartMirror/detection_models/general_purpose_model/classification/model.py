"""
Classification Model Module

Defines the Temporal Convolutional Network (TCN) architecture for exercise type classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, metrics
from tensorflow.keras.losses import SparseCategoricalCrossentropy
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


def build_exercise_classification_model(
    input_dim,
    num_classes,
    num_filters=64,
    kernel_size=3,
    num_layers=4,
    dropout_rate=0.2,
    learning_rate=1e-3
):
    """
    Builds and compiles a TCN-based model for exercise type classification.
    
    The model predicts exercise type using:
    - Stacked residual TCN blocks with exponential dilation
    - Global average pooling to aggregate temporal information
    - Dense layers for final classification
    - Softmax activation for multi-class output
    
    Args:
        input_dim: Number of input features per frame (typically 25 joint angles)
        num_classes: Number of exercise classes to predict
        num_filters: Number of filters in TCN layers
        kernel_size: Size of the convolution kernel
        num_layers: Number of TCN layers
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model for exercise classification
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

    # Global average pooling to aggregate temporal information
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers for classification
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Final classification layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create model
    model = models.Model(inputs, outputs, name='tcn_classifier')

    # Compile with categorical crossentropy for multi-class classification
    loss = SparseCategoricalCrossentropy(from_logits=False)
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            metrics.SparseCategoricalAccuracy(name='accuracy'),
            metrics.SparseCategoricalCrossentropy(name='crossentropy')
        ]
    )
    
    return model


def get_model_summary(input_dim=25, num_classes=4):
    """
    Get a summary of the model architecture.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of exercise classes
        
    Returns:
        Model summary string
    """
    model = build_exercise_classification_model(
        input_dim=input_dim, 
        num_classes=num_classes
    )
    return model.summary()


def save_model_plot(input_dim=25, num_classes=4, output_path='tcn_classifier.png'):
    """
    Save a visual plot of the model architecture.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of exercise classes
        output_path: Path to save the plot
    """
    model = build_exercise_classification_model(
        input_dim=input_dim, 
        num_classes=num_classes
    )
    plot_model(
        model, 
        to_file=output_path, 
        show_shapes=True, 
        expand_nested=True
    )
    print(f"Model architecture plot saved to: {output_path}")


if __name__ == '__main__':
    """Example usage and model visualization."""
    print("Building exercise classification model...")
    
    # Create model
    model = build_exercise_classification_model(
        input_dim=25, 
        num_classes=4  # push-ups, squats, pull-ups, dips
    )
    
    # Print summary
    print("\nModel Summary:")
    model.summary()
    
    # Save architecture plot
    save_model_plot(
        input_dim=25, 
        num_classes=4, 
        output_path='tcn_classifier.png'
    )
    
    print("\nModel created successfully!") 