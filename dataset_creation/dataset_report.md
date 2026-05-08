# Dataset Report: Custom Student Engagement (Room-Specific)

## 1. Overview
This dataset was specifically developed to enhance the performance of the Student Engagement ViT (Vision Transformer) model. It focuses on closing the **Domain Gap** between generic training data and the specific environmental conditions (lighting, background, and camera angle) of the live inference setup.

## 2. Dataset Structure
The dataset is organized into three primary classes corresponding to different levels of student engagement:

- **Highly Engaged**: Focused posture, direct eye contact with the screen, active participation.
- **Moderately Engaged**: Occasional distractions, varied posture, but generally attentive.
- **Disengaged**: Slumping, looking away, phone usage, or absence of screen interaction.

### File Organization
```text
engagement_dataset/
├── highly_engaged/        (348 images)
├── moderately_engaged/    (389 images)
└── disengaged/           (387 images)
```

## 3. Data Acquisition & Processing
The data was sourced from raw `.webm` video recordings captured in the target environment.

### Methodology:
- **Frame Extraction**: Images were extracted using a sampling rate of every **5th frame** to ensure a high diversity of poses while minimizing near-identical frames.
- **Image Format**: All frames are saved as high-quality `.jpg` files.
- **Naming Convention**: Files follow a clean sequential format: `{class_name}_{index:03d}.jpg` (e.g., `highly_engaged_042.jpg`).
- **Standardization**: All images are stored in class-specific subdirectories for direct compatibility with standard deep learning data loaders (e.g., PyTorch `ImageFolder`).

## 4. Key Statistics
| Metric | Value |
| :--- | :--- |
| **Total Images** | 1,124 |
| **Class Distribution** | Balanced (Approx. 31-35% per class) |
| **Resolution** | 720p/1080p (Source Video Native) |
| **Target Model** | Swin Transformer / E-ViT |

## 5. Technical Significance
By introducing this custom data, the model's "live" performance is significantly improved. Specifically, it prevents the model from misinterpreting static background elements (like furniture or wall decor) as "Disengaged" features, as these elements are now present across all training labels.

## 6. Usage Instructions
1. Upload the `engagement_dataset/` folder to Kaggle.
2. Merge with the original Student Engagement dataset in your training script.
3. Re-train the Swin Transformer to achieve optimal live inference results.
