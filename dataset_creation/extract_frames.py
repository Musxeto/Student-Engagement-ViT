import cv2
import os

# --- CONFIGURATION ---
# Base directory is where this script is located
base_dir = os.path.dirname(os.path.abspath(__file__))
video_dir = os.path.join(base_dir, 'videos')
output_dir = os.path.join(base_dir, 'engagement_dataset')
classes = ['highly_engaged', 'moderately_engaged', 'disengaged']
frame_rate = 5 # Capture every 10th frame to avoid identical images

def extract_frames():
    print(f"Looking for videos in: {video_dir}")
    print(f"Saving frames to: {output_dir}")
    
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for label in classes:
        class_folder = os.path.join(output_dir, label)
        os.makedirs(class_folder, exist_ok=True)
        
        # Find all videos for this class
        videos = [v for v in os.listdir(video_dir) if v.startswith(label)]
        
        if not videos:
            print(f"No videos found for class: {label}")
            continue

        global_saved_count = 1
        for v_name in videos:
            video_path = os.path.join(video_dir, v_name)
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                continue
                
            count = 0
            video_saved_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if count % frame_rate == 0:
                    # Clean naming: class_name_index.jpg (with leading zeros for better sorting)
                    img_name = f"{label}_{global_saved_count:03d}.jpg"
                    cv2.imwrite(os.path.join(class_folder, img_name), frame)
                    global_saved_count += 1
                    video_saved_count += 1
                
                count += 1
            
            cap.release()
            print(f"Extracted {video_saved_count} frames from {v_name}")

    print(f"\nDone! Your custom dataset is ready in: {output_dir}")

if __name__ == "__main__":
    extract_frames()
