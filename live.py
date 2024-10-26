import face_recognition
import cv2
import numpy as np
import dlib
from mtcnn import MTCNN
import os
from PIL import Image

# Paths to resources and output folders
unique_faces_dir = r'C:\Users\parab\OneDrive\Desktop\EBAA\Unique_faces'
weights_path = r'C:\Users\parab\OneDrive\Desktop\EBAA\yolov3.weights'
config_path = r'C:\Users\parab\OneDrive\Desktop\EBAA\yolov3.cfg'
labels_path = r'C:\Users\parab\OneDrive\Desktop\EBAA\coco.names'
frames_dir = r'C:\Users\parab\OneDrive\Desktop\EBAA\output_frames'
predictor_path = r'C:\Users\parab\OneDrive\Desktop\EBAA\shape_predictor_68_face_landmarks.dat'
pose_txt_path = r'C:\Users\parab\OneDrive\Desktop\EBAA\face_pose_angles.txt'

# Check if YOLO files exist
if not os.path.exists(weights_path) or not os.path.exists(config_path):
    print("YOLOv3 files not found.")
    exit()

# Create directory for processed frames if it doesn’t exist
os.makedirs(frames_dir, exist_ok=True)

# Load YOLO model
labels = open(labels_path).read().strip().split("\n")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# MTCNN and Dlib setup
mtcnn_detector = MTCNN()
dlib_predictor = dlib.shape_predictor(predictor_path)

# Camera matrix and distortion coefficients
focal_length = 640
center = (320, 240)
camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

# 3D model points of facial landmarks
model_points = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-165.0, 170.0, -135.0),    # Left eye left corner
    (165.0, 170.0, -135.0),     # Right eye right corner
    (-150.0, -150.0, -125.0),   # Left mouth corner
    (150.0, -150.0, -125.0)     # Right mouth corner
])

def showLiveFeed(frame_placeholder, run_live_feed):
    cap = cv2.VideoCapture(0)
    face_id = 0
    frame_counter = 0
    humans_looking = 0
    face_nums_pos_angles = {}

    while run_live_feed:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

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
                if confidence > 0.5 and labels[classID] == 'person':
                    box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    centerX, centerY, width, height = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))

        # Apply Non-Maximum Suppression (NMS)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

        # Face detection and recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = mtcnn_detector.detect_faces(rgb_frame)
        for i, face in enumerate(faces):
            bbox = face['box']
            x, y, w, h = bbox
            dlib_rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)
            landmarks = dlib_predictor(rgb_frame, dlib_rect)
            image_points = np.array([
                (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
                (landmarks.part(8).x, landmarks.part(8).y),   # Chin
                (landmarks.part(36).x, landmarks.part(36).y), # Left eye left corner
                (landmarks.part(45).x, landmarks.part(45).y), # Right eye right corner
                (landmarks.part(48).x, landmarks.part(48).y), # Left mouth corner
                (landmarks.part(54).x, landmarks.part(54).y)  # Right mouth corner
            ], dtype=np.float32)

            # Calculate pose
success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
yaw, pitch, roll = get_euler_angles(rotation_matrix)

face_nums_pos_angles[name][0] += abs(yaw)
face_nums_pos_angles[name][1] += abs(pitch)
face_nums_pos_angles[name][2] += abs(roll)
face_nums_pos_angles[name][3] += 1

if face_nums_pos_angles[name][3] == 3:
    avg_yaw = face_nums_pos_angles[name][0] / face_nums_pos_angles[name][3]
    avg_pitch = face_nums_pos_angles[name][1] / face_nums_pos_angles[name][3]
    avg_roll = face_nums_pos_angles[name][2] / face_nums_pos_angles[name][3]

    if avg_yaw > 139 and avg_yaw < 175:
        humans_looking += 1

    print(f"Humans looking are {humans_looking}")
    print(f"Total humans are {name}")

# Draw rectangles and labels on faces
cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Save the frame in the frames directory if faces are detected
if faces:
    frame_filename = os.path.join(frames_dir, f"frame_{frame_counter}.jpg")
    cv2.imwrite(frame_filename, frame)

# Display the frame
# cv2.imshow('Face Recognition', frame)

frame_counter += 1
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Convert to PIL Image
img = Image.fromarray(frame_rgb)

# Update the placeholder
frame_placeholder.image(img, channels="RGB")

# Press 'q' to exit
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break

# Release video capture and close all windows
cap.release()
# cv2.destroyAllWindows()
