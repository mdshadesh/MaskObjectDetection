import cv2

# Load the pre-trained cascade classifiers for face and mask detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mask_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mask.xml')

# Load the input image or use the webcam
# Replace 'path_to_image' with the actual path or leave it blank to use the webcam
image_path = 'path_to_image'  # Leave it blank to use the webcam

if image_path:
    # Read the input image from file
    image = cv2.imread(image_path)
else:
    # Capture video from the webcam
    video_capture = cv2.VideoCapture(0)

while True:
    if image_path:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(image if image_path else frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the region of interest (ROI) for face detection of masks
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        # Detect masks in the ROI
        masks = mask_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        # If no masks are detected, display "No Mask"
        if len(masks) == 0:
            cv2.putText(image if image_path else frame, 'No Mask', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            # Draw a rectangle around the masks
            for (mx, my, mw, mh) in masks:
                cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (255, 0, 0), 2)
                cv2.putText(image if image_path else frame, 'Mask', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if image_path:
        # Display the output image with face and mask detections
        cv2.imshow('Face Mask Detection', image)
        cv2.waitKey(0)
        break
    else:
        # Display the resulting frame from webcam
        cv2.imshow('Face Mask Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close all windows
if not image_path:
    video_capture.release()
cv2.destroyAllWindows()
