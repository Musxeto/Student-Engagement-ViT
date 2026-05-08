import cv2
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from transformers import SwinConfig, SwinForImageClassification

# 1. Configuration
MODEL_PATH = "best_swin_model.pth"
# Replace this with your IP Camera URL or use 0 for local webcam
IP_CAMERA_URL = "https://192.168.100.118:8080/video" 

# Class names mapping from your training
CLASS_NAMES = ['Disengaged', 'Highly_Engaged', 'Moderately_Engaged']
COLORS = [(0, 0, 255), (0, 255, 0), (0, 255, 255)] # Red, Green, Yellow (BGR format)

def main():
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

    # 4. Start Video Capture
    print(f"Connecting to camera: {IP_CAMERA_URL}")
    cap = cv2.VideoCapture(IP_CAMERA_URL)

    if not cap.isOpened():
        print(f"Warning: Could not open IP camera at {IP_CAMERA_URL}")
        print("Attempting to use local webcam (0) instead...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open local webcam. Please check your camera connection.")
            return

    print("Starting student-level inference. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Reconnecting...")
            cv2.waitKey(1000)
            continue

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        # scaleFactor and minNeighbors can be adjusted for sensitivity
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

        if len(faces) == 0:
            # Show the original frame if no one is detected
            cv2.putText(frame, "No students detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            for (x, y, w, h) in faces:
                # Add some padding around the face to include head/shoulders (helps classification)
                padding = int(w * 0.3)
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)
                
                # Crop and convert to PIL
                crop = frame[y1:y2, x1:x2]
                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_crop)

                # Preprocess and Inference
                input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                    
                    label = CLASS_NAMES[predicted_idx.item()]
                    score = confidence.item()

                # Display Results on Frame
                color = COLORS[predicted_idx.item()]
                display_text = f"{label}: {score:.0%}"
                
                # Draw Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw Label Background and Text
                label_size, base_line = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                y1_label = max(y1, label_size[1] + 10)
                cv2.rectangle(frame, (x1, y1_label - label_size[1] - 10), (x1 + label_size[0], y1_label), color, -1)
                cv2.putText(frame, display_text, (x1, y1_label - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Student Engagement Monitoring', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
