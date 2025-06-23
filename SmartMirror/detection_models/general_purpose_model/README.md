# Exercise Detection System

A machine learning system for detecting and analyzing exercise repetitions using computer vision and deep learning.

## Features

- **Segmentation Models**: Detect individual exercise repetitions in video sequences
- **Classification Models**: Classify exercise types (planned for future)
- **Real-time Processing**: Process video streams with MediaPipe pose estimation
- **Interactive Labeling**: Manual video labeling tool for training data
- **Comprehensive CLI**: Easy-to-use command-line interface for all operations

## Project Structure

```
general_purpose_model/
├── 📁 data/                    # Data storage
│   ├── 📁 raw/                 # Raw video files
│   ├── 📁 labels/              # Manual labels (CSV files)
│   └── 📁 processed/           # Processed datasets (NPZ files)
├── 📁 models/                  # Trained models
│   ├── 📁 segmentation/        # Segmentation models
│   └── 📁 classification/      # Classification models (future)
├── 📁 segmentation/            # Segmentation model code
│   ├── 📄 model.py            # TCN-based segmentation model
│   └── 📄 trainer.py          # Training script
├── 📁 classification/          # Classification model code (future)
├── 📁 output_videos/           # Generated videos and plots
├── 📁 test_videos/             # Test video files
├── 📄 main.py                  # Main CLI interface
├── 📄 dataset_builder.py       # Dataset creation and processing
├── 📄 video_labeler.py         # Manual labeling interface
├── 📄 segmentation_demo.py     # Segmentation testing demo
├── 📄 system_test.py           # System testing script
├── 📄 requirements.txt         # Python dependencies
└── 📄 README.md               # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test the System

```bash
python system_test.py
```

### 3. Label Videos (Optional)

```bash
python main.py label --exercise push-ups
```

### 4. Build Dataset

```bash
python main.py build-dataset --mode segmentation --exercise push-ups
```

### 5. Train Model

```bash
python main.py train --mode segmentation --exercise push-ups
```

### 6. Test Model

```bash
python main.py test --mode segmentation --input test_videos/video.mp4 --model push-ups
```

## Usage Examples

### Building Datasets

```bash
# Build segmentation dataset for push-ups
python main.py build-dataset --mode segmentation --exercise push-ups

# Build classification dataset (all exercises)
python main.py build-dataset --mode classification
```

### Training Models

```bash
# Train segmentation model for push-ups
python main.py train --mode segmentation --exercise push-ups --epochs 1000

# Train with custom parameters
python main.py train --mode segmentation --exercise squats --batch 32 --lr 0.001
```

### Testing Models

```bash
# Test segmentation model
python main.py test --mode segmentation --input test_videos/pushup.mp4 --model push-ups

# Test with custom parameters
python main.py test --mode segmentation --input test_videos/squat.mp4 --model squats --smooth 20 --prom 0.2
```

### Interactive Labeling

```bash
# Label push-up videos
python main.py label --exercise push-ups
```

## Model Architecture

### Segmentation Model
- **Architecture**: Temporal Convolutional Network (TCN)
- **Input**: Pose angle features (25 dimensions)
- **Output**: Repetition probability per frame
- **Window Size**: 140 frames (configurable)
- **Features**: MediaPipe pose angles

### Classification Model (Future)
- **Architecture**: TBD
- **Input**: Pose features
- **Output**: Exercise type classification
- **Classes**: push-ups, squats, pull-ups, dips

## Data Format

### Raw Videos
- Format: MP4
- Location: `data/raw/{exercise_type}/`
- Naming: `{exercise_type}_{number}.mp4`

### Labels
- Format: CSV
- Location: `data/labels/{exercise_type}/`
- Columns: frame_number, label (0 or 1)

### Processed Datasets
- Format: NPZ (NumPy compressed)
- Location: `data/processed/`
- Content: Features (X) and labels (y)

## Configuration

### Training Parameters
- **Window Size**: 140 frames (default)
- **Batch Size**: 16 (default)
- **Learning Rate**: 1e-4 (default)
- **Epochs**: 10000 (default)

### Peak Detection Parameters
- **Smoothing Window**: 15 frames (default)
- **Prominence**: 0.15 (default)
- **Distance**: 25 frames (default)

## Output Files

### Annotated Videos
- Location: `output_videos/annotated/`
- Format: MP4 with overlay text
- Content: Probability, rep count, timestamp

### Plots
- **Probability Plot**: `output_videos/plots/`
- **Peak Detection Plot**: `output_videos/peak_detection/`

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're using the correct Python environment
2. **Model Not Found**: Check that models exist in `models/segmentation/`
3. **Video Processing Errors**: Ensure videos are in MP4 format
4. **Memory Issues**: Reduce batch size or window size

### Testing

Run the system test to verify everything is working:

```bash
python system_test.py
```

## Development

### Adding New Exercises

1. Add videos to `data/raw/{new_exercise}/`
2. Label videos using `python main.py label --exercise {new_exercise}`
3. Build dataset: `python main.py build-dataset --mode segmentation --exercise {new_exercise}`
4. Train model: `python main.py train --mode segmentation --exercise {new_exercise}`

### Extending the System

- **New Features**: Add to `FeatureExtractor` class
- **New Models**: Create in `segmentation/` or `classification/` folders
- **New Datasets**: Extend `DatasetBuilder` classes

## License

This project is part of the SmartMirror system. 