import os
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms, datasets
from facenet_pytorch import MTCNN
from model import EthnicityModel
from datetime import datetime

# Setup device and detector
device = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(min_face_size=50,thresholds=[0.8, 0.9, 0.9],keep_all=False, device=device)

def classify_ethnicity(image_path, model, ethnicity_labels, face_size_threshold=20, device=None, save_dir="detected_images"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        img = cv2.imread(image_path)
        if img is None:
            return "Error: Could not read image"

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect face
        try:
            boxes, probs, landmarks = detector.detect(rgb_img, landmarks=True)
        except Exception as e:
            return f"Error in MTCNN detection: {e}"

        if boxes is None or len(boxes) == 0:
            return "No face detected in the image"

        best_idx = int(np.argmax(probs))
        score = probs[best_idx]
        box = boxes[best_idx]

        if score < 0.95:
            return f"Low confidence detection: {score:.2f}"

        x1, y1, x2, y2 = map(int, box)
        width, height = x2 - x1, y2 - y1
        if width < face_size_threshold or height < face_size_threshold:
            return f"Detected face too small: {width}x{height}"

        # Visualization - Save to `save_dir`
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        vis_filename = f"{timestamp}_detected_face.jpg"
        vis_path = os.path.join(save_dir, vis_filename)
        vis_img = img.copy()
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(vis_path, vis_img)

        # Crop & resize
        margin = 15
        x1m = max(0, x1 - margin)
        y1m = max(0, y1 - margin)
        x2m = min(img.shape[1], x2 + margin)
        y2m = min(img.shape[0], y2 + margin)
        face_img = img[y1m:y2m, x1m:x2m]
        face_img = cv2.resize(face_img, (128, 128))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # To tensor
        face_pil = Image.fromarray(face_img)
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        face_tensor = val_transform(face_pil).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(face_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            confidence = confidence.item()
            predicted_class = predicted_class.item()

            all_probabilities = {
                ethnicity_labels.get(idx, f"class_{idx}"): float(prob)
                for idx, prob in enumerate(probabilities.squeeze().tolist())
            }

        ethnicity = ethnicity_labels.get(predicted_class, "Unknown")
        return {
            "ethnicity": ethnicity,
            "confidence": confidence,
            "face_confidence": float(score),
            "face_size": f"{width}x{height}",
            "visualization_saved": vis_path,
            "all_probabilities": all_probabilities
        }

    except Exception as e:
        return f"Error processing image: {str(e)}"


# # ---------- Main Inference Runner ----------
#
# DATASET_PATH = "D:/Projects/Asian Face Recognition/combined_dataset"
# MODEL_PATH = "ethnicity_model.pth"
# LABELS_PATH = "ethnicity_labels.npy"
#
# # Load label map
# full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=None)
# class_to_idx = full_dataset.class_to_idx
# ethnicity_labels = {idx: label for label, idx in class_to_idx.items()}
#
# # Load model
# num_classes = len(ethnicity_labels)
# model = EthnicityModel(num_classes)
#
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.to(device)
#
# # Test Image
# test_image_path = "C:/Users/Administrator/Downloads/blue background - Faizan Ali.jpeg"
# result = classify_ethnicity(test_image_path, model, ethnicity_labels, device=device)
# print(result)
