import cv2
import numpy as np
import os

# Paths to resources and output folders
unique_faces_dir = r'C:\Users\parab\OneDrive\Desktop\EBAA\Unique_faces'
weights_path = r'C:\Users\parab\OneDrive\Desktop\EBAA\yolov3.weights'
config_path = r'C:\Users\parab\OneDrive\Desktop\EBAA\yolov3.cfg'
labels_path = r'C:\Users\parab\OneDrive\Desktop\EBAA\coco.names'
frames_dir = r'C:\Users\parab\OneDrive\Desktop\EBAA\output_frames'

# Check if YOLO files exist
if not os.path.exists(weights_path) or not os.path.exists(config_path):
    print("YOLOv3 files not found.")
    exit()

# Create directory for processed frames if it doesn’t exist
os.makedirs(frames_dir, exist_ok=True)

labels = open(labels_path).read().strip().split("\n")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# YOLO confidence threshold
confidence_threshold = 0.5
nms_threshold = 0.3

def showPreProcessedVideo(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame or end of video.")
            break

        total_frames += 1

        # YOLO detection for person count
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(net.getUnconnectedOutLayersNames())
        boxes = []
        confidences = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confidence_threshold and labels[classID] == 'person':
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    centerX, centerY, width, height = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))

        # Apply NMS
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

        # Ensure idxs is not empty and extract valid indices
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y, w, h) = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # Count true positives
                true_positives += 1
        else:
            false_negatives += 1

        # Calculate accuracy metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = true_positives / (true_positives + false_positives + false_negatives) if (true_positives + false_positives + false_negatives) > 0 else 0

        # Convert to percentage
        precision_percentage = precision * 100
        recall_percentage = recall * 100
        f1_percentage = f1_score * 100
        accuracy_percentage = accuracy * 100

        # Print detailed output
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")
        print(f"Total Frames: {total_frames}")
        print(f"Estimated Precision: {precision_percentage:.2f}%, Estimated Recall: {recall_percentage:.2f}%, Estimated F1 Score: {f1_percentage:.2f}%, Estimated Accuracy: {accuracy_percentage:.2f}%")

    cap.release()

# Usage
video_path = r'C:\Users\parab\OneDrive\Desktop\EBAA\Input_frames\karan.mp4'
showPreProcessedVideo(video_path)
