# 🎓 Student Engagement Recognition using Swin Transformers

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange?logo=huggingface&logoColor=white)](https://huggingface.co/docs/transformers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art computer vision system designed to monitor and classify student engagement levels in real-time during online educational sessions. Leveraging the power of **Swin Transformers (Shifted Window Transformers)** and **OpenCV**, this project provides a robust solution for understanding student behavior through automated facial analysis.

---

## 🌟 Key Features

-   **🎯 High-Precision Classification**: Utilizes `Swin-Base` architecture for superior feature extraction compared to traditional CNNs.
-   **⏱️ Real-Time Monitoring**: Supports live inference from webcams or IP cameras with optimized frame processing.
-   **📂 Video Analytics**: Batch processing for recorded sessions with automated output rendering and progress tracking.
-   **🎭 Smart Face Detection**: Integrated OpenCV Haar Cascades for dynamic student tracking and localized engagement analysis.
-   **📊 3-Level Engagement Scale**:
    -   🟢 **Highly Engaged**: Active participation and focus.
    -   🟡 **Moderately Engaged**: Passive listening/observing.
    -   🔴 **Disengaged**: Lack of focus or absence from the frame.

---

## 🛠️ Tech Stack

-   **Deep Learning**: PyTorch, Hugging Face Transformers (Swin-T)
-   **Computer Vision**: OpenCV
-   **Data Handling**: PIL, NumPy
-   **UI/Progress**: TQDM

---

## 🚀 Getting Started

### Prerequisites

-   Python 3.8 or higher
-   CUDA-enabled GPU (Highly recommended for real-time inference)

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/YourUsername/Student-Engagement-ViT.git
    cd Student-Engagement-ViT
    ```

2.  **Set Up Environment**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Model Weights**
    Ensure `best_swin_model.pth` is placed in the root directory.

---

## 💻 Usage

### 1. Live Inference
Monitor student engagement in real-time via a webcam or an IP camera.

```bash
python live_inference.py
```
*Tip: Update `IP_CAMERA_URL` in the script to point to your camera stream.*

### 2. Video Inference
Process a recorded video file and save the annotated results.

```bash
python video_inference.py
```
*Configuration: Modify `INPUT_VIDEO` and `OUTPUT_VIDEO` in the script as needed.*

---

## 📁 Project Structure

```text
Student-Engagement-ViT/
├── best_swin_model.pth       # Trained Swin Transformer weights
├── live_inference.py         # Real-time webcam/IP camera script
├── video_inference.py        # Recorded video processing script
├── student-engagement-vit.ipynb # Training & Evaluation notebook
├── dataset_creation/         # Scripts for data preparation
├── Student-engagement-dataset/ # Dataset storage
├── results/                  # Inference output directory
└── requirements.txt          # Python dependencies
```

---

## 📈 Model Performance

The system is trained on a curated student engagement dataset with a focus on overcoming class imbalance through weighted loss and augmentation.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 92.4% |
| **Macro F1-Score** | 0.89 |
| **Inference Latency** | ~45ms (RTX 3060) |

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git checkout -b feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<p align="center">
  Developed with ❤️ for the Computer Vision Course
</p>
