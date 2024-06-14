import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from IPython.display import display, Image, clear_output

# Load the model
from keras.models import load_model
model = load_model("emotiondetector.keras", compile=False)

# Compile the model with the same optimizer
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Haarcascade file for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess the image
def extract_features(image):
    feature = img_to_array(image)
    feature = feature.reshape(1, 48, 48, 1)
    feature = feature.astype('float32') / 255.0
    return feature

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not webcam.isOpened():
    print("Error: Could not open webcam.")
else:
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face = cv2.resize(face, (48, 48))
            face = extract_features(face)
            pred = model.predict(face)
            prediction_label = labels[np.argmax(pred)]
            cv2.putText(frame, f'{prediction_label}', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        # Display the processed frame in a separate window
        cv2.imshow('Emotion Detector', frame)

        # Check for the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    webcam.release()
    cv2.destroyAllWindows()