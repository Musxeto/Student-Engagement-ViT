
Vision Transformer-Based Student Engagement Recognition in Online Education - Computer Vision Course Semester Project

## Project Overview
This project implements a student engagement recognition system using Vision Transformers (ViT) to classify student engagement levels in online educational environments. The system is trained and evaluated on the "Student-engagement-dataset", handling the challenges ofimbalanced class distributions and varying image quality.

## Getting Started

### Prerequisites
- Python 3.6+
- pip

### Installation
1. Clone the repository (or download the project files).
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation
1. Ensure the dataset is placed in the `Student-engagement-dataset/` directory (or update the paths in the scripts).
2. Run the data preprocessing script to prepare the data for training:
   ```bash
   python src/data_loader.py
   ```
   This script will create `train.csv` and `test.csv` in the `dataset/` directory.

### Training
Train the ViT model using the prepared dataset:

```bash
python train.py
```

**Configuration:**
- The default configuration is set to use `hybrid_transformer.yaml`.
- You can customize hyperparameters, model architecture, and training settings in `src/configs/hybrid_transformer.yaml`.

### Evaluation
Evaluate the trained model on the test set:

```bash
python evaluate.py
```

This will generate performance metrics and save the results to the `__results___files/` directory.

## Code Structure

```
Student-Engagement-ViT/
├── dataset/                  # Processed dataset and CSV files
├── Student-engagement-dataset/  # Original dataset
├── src/
│   ├── configs/             # Configuration files (YAML)
│   ├── data_loader.py       # Data loading and preprocessing
│   └── vit_model.py         # Vision Transformer model implementation
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
└── requirements.txt         # Project dependencies
```

## Key Features

### 1. Hybrid Architecture
This project implements a **Hybrid Student Engagement Recognition Model** that combines:
- **Vision Transformer (ViT)**: For global feature extraction and capturing long-range dependencies.
- **Convolutional Neural Network (CNN)**: For local feature extraction and capturing fine-grained details.
- **Attention Mechanism**: To focus on the most informative regions of the input image.

### 2. Handling Class Imbalance
The dataset has a significant class imbalance. This project addresses this using:
- **Weighted Loss**: Assigning higher weights to underrepresented classes during training.
- **Data Augmentation**: To increase the effective size of the minority classes.
- **Proper Evaluation Metrics**: Using F1-score, precision, and recall in addition to accuracy.

## Configuration
Model hyperparameters and training settings are managed through YAML configuration files in `src/configs/`. The default configuration is `hybrid_transformer.yaml`.

You can modify:
- `model.architecture`: To switch between 'vit', 'cnn', or 'hybrid'.
- `training.learning_rate`, `training.batch_size`, `training.epochs`: For training parameters.
- `data.path`: To change the dataset location.

## Output
- **Training**: Training logs and model checkpoints are saved to `__logs__files/`.
- **Evaluation**: Performance metrics, confusion matrix, and classification report are saved to `__results___files/`.

## Troubleshooting
- **Memory Errors**: Reduce `training.batch_size` in the config file if you encounter CUDA memory errors.
- **Data Loading Issues**: Ensure the dataset is correctly placed in `Student-engagement-dataset/` and run `python src/data_loader.py` first.
- **Missing Dependencies**: Run `pip install -r requirements.txt` to install all required packages.
