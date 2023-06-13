import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# Load the pre-trained MobileNetV2 model
model = load_model('path_to_model')  # Replace 'path_to_model' with the actual path to your trained model

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the labels for mask and no mask
labels = {0: 'Mask', 1: 'No Mask'}

# Define the color for mask and no mask bounding boxes
colors = {0: (0, 255, 0), 1: (0, 0, 255)}

# Open video capture
video_capture = cv2.VideoCapture('path_to_video')  # Replace 'path_to_video' with the actual path to your video file

while True:
    # Read frame from video
    ret, frame = video_capture.read()
    
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face region from the frame
        face_img = frame[y:y+h, x:x+w].copy()

        # Preprocess the face image for the CNN model
        face_img = cv2.resize(face_img, (224, 224))
        face_img = preprocess_input(face_img)
        face_img = np.expand_dims(face_img, axis=0)

        # Make predictions using the trained model
        predictions = model.predict(face_img)
        label_index = np.argmax(predictions)
        label = labels[label_index]
        color = colors[label_index]

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the resulting frame
    cv2.imshow('Face Mask Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
