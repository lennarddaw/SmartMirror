"""
visualize_dataset.py

Script zum Laden einer .npz-Dataset-Datei (Frame-basiert) und Visualisieren jeder Feature-Dimension in einem kombinierten Diagramm mit Hervorhebung des Outputs.
"""
import numpy as np
import matplotlib.pyplot as plt
import os


def load_dataset(path: str = 'dataset.npz') -> tuple[np.ndarray, np.ndarray]:
    """
    Lädt X- und y-Arrays aus einer NPZ-Datei.
    Erwartet X mit Form (num_frames, num_features) und y mit Form (num_frames,).
    """
    data = np.load(path)
    X = data['X']  # (num_frames, num_features)
    y = data['y']  # (num_frames,)
    return X, y


def plot_features_with_output(
    X: np.ndarray,
    y: np.ndarray,
    save_path: str | None = None
):
    """
    Zeichnet alle Features über die ersten 1000 Frames in verschiedenen Stilen:
    - Features 5 und 6 in Vordergrund (fette Linien)
    - Andere Features im Hintergrund (grau, transparent)
    Hebt außerdem den Output (y) als dünne schwarze Linie hervor.

    Args:
        X: Feature-Array der Form (num_frames, num_features)
        y: Label-Array der Form (num_frames,)
        save_path: Optionaler Pfad zum Speichern des Diagramms (PNG)
    """
    # Beschränke Daten auf die ersten 1000 Frames
    max_frames = min(1000, X.shape[0])
    X = X[:max_frames]
    y = y[:max_frames]
    num_frames, num_features = X.shape

    plt.figure(figsize=(14, 6))
    t = np.arange(num_frames)

    # Hintergrund-Features (alle außer 5 und 6)
    for feat in range(num_features):
        if feat not in (3, 4, 5, 6):
            plt.plot(
                t,
                X[:, feat],
                color='gray',
                alpha=0.3,
                linewidth=0.8
            )

    # Hervorgehobene Features
    for feat in (3, 4, 5, 6):
        plt.plot(
            t,
            X[:, feat],
            linewidth=2,
            label=f'Feature {feat}'
        )

    # Label als dünne schwarze Linie oberhalb
    offset = 1.1 * np.max(np.abs(X))
    plt.step(
        t,
        y * offset + offset,
        where='mid',
        linewidth=1,
        label='Output (Label)',
        color='black'
    )

    plt.title('Frame-basierte Features & Output')
    plt.xlabel('Frame Index')
    plt.ylabel('Winkel (normiert)')
    plt.legend(loc='upper right', ncol=2, fontsize='small')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path)
        print(f'[INFO] Saved combined plot to {save_path}')
    else:
        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize frame-based features in one plot with highlighted output'
    )
    parser.add_argument(
        '--input', type=str, default='dataset.npz', help='Pfad zur NPZ-Datei'
    )
    parser.add_argument(
        '--save', type=str, default=None, help='Pfad zum Speichern des PNG (optional)'
    )
    args = parser.parse_args()

    X, y = load_dataset(args.input)
    print(f'Loaded dataset: X={X.shape}, y={y.shape}')

    plot_features_with_output(
        X,
        y,
        save_path=args.save
    )