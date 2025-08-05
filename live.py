import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json

# Load the model
model = load_model('emotion_model.h5')

# Load class labels
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]

        # Preprocess face
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=(0, -1))  # (1,48,48,1)

        # Predict emotion
        preds = model.predict(face_input, verbose=0)
        class_index = np.argmax(preds)
        emotion = class_labels[class_index]
        confidence = np.max(preds)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f'{emotion} ({confidence*100:.1f}%)', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    # Show frame
    cv2.imshow('Emotion Detection', frame)

    # Break loop on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
