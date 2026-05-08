import cv2
import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from transformers import SwinConfig, SwinForImageClassification
from tqdm import tqdm

# 1. Configuration
MODEL_PATH = "best_swin_model.pth"
INPUT_VIDEO = "test_data/zoom_meeting.mp4"
OUTPUT_FOLDER = "results"
OUTPUT_VIDEO = os.path.join(OUTPUT_FOLDER, "zoom_meeting_processed.mp4")

# Class names mapping from your training
CLASS_NAMES = ['Disengaged', 'Highly_Engaged', 'Moderately_Engaged']
COLORS = [(0, 0, 255), (0, 255, 0), (0, 255, 255)] # Red, Green, Yellow (BGR format)

def main():
    # Create results directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 2. Load Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Face Detector (Haar Cascade)
    print("Initializing face detector...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Loading model architecture from config...")
    try:
        config = SwinConfig.from_pretrained(
            "microsoft/swin-base-patch4-window7-224-in22k", 
            num_labels=3
        )
        model = SwinForImageClassification(config)

        print(f"Loading local weights from {MODEL_PATH}...")
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Define Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Initialize Video Capture
    if not os.path.exists(INPUT_VIDEO):
        print(f"Error: Input video file not found at {INPUT_VIDEO}")
        return

    cap = cv2.VideoCapture(INPUT_VIDEO)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize Video Writer (try avc1 for better compatibility, fall back to mp4v)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Warning: avc1 codec failed. Falling back to mp4v...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print(f"Processing video: {INPUT_VIDEO} ({total_frames} frames)")
    print(f"Saving output to: {OUTPUT_VIDEO}")

    # Use tqdm for progress bar
    pbar = tqdm(total=total_frames, desc="Inference Progress")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

            for (x, y, w, h) in faces:
                # Add padding around the face
                padding = int(w * 0.3)
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)
                
                # Crop and predict
                try:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    
                    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_crop)

                    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                        confidence, predicted_idx = torch.max(probabilities, 1)
                        
                        label = CLASS_NAMES[predicted_idx.item()]
                        score = confidence.item()

                    # Draw Results
                    color = COLORS[predicted_idx.item()]
                    display_text = f"{label}: {score:.0%}"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    label_size, _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    y1_label = max(y1, label_size[1] + 10)
                    cv2.rectangle(frame, (x1, y1_label - label_size[1] - 10), (x1 + label_size[0], y1_label), color, -1)
                    cv2.putText(frame, display_text, (x1, y1_label - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.8, (0, 0, 0), 2, cv2.LINE_AA)
                except Exception:
                    continue

            # Write frame to output video
            out.write(frame)

            # Optional: Show preview (press 'q' to stop)
            cv2.imshow('Video Inference Preview', cv2.resize(frame, (1280, 720)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nProcessing stopped by 'q' key.")
                break
            
            pbar.update(1)

    except KeyboardInterrupt:
        print("\nProcessing interrupted by Ctrl+C.")
    finally:
        pbar.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"\nProcessing stopped. Progress saved to: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
