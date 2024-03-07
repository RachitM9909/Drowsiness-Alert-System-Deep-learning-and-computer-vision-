
import cv2
from keras.models import load_model
import numpy as np
from playsound import playsound

# Load the pre-trained model
model = load_model('C:/Users/mistr/jupyter notebook codes/drowsiness detection/models/drowsinessmodel.h5')


# Define function to preprocess image for prediction
def preprocess_image(img):
    img = cv2.resize(img, (256, 256))  # Resize image to match model's expected sizing
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values
    return img


# Function to predict drowsiness and make a beep sound
def predict_drowsiness(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    return prediction


# Function to generate a beep sound
def beep():
    playsound("C:/Users/mistr/Downloads/beep-01a.mp3")


# Initialize webcam
cap = cv2.VideoCapture(1)  # Change the argument if using a different camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform prediction on the frame
    result = predict_drowsiness(frame)

    # Display result
    if result < 0.5:
        cv2.putText(frame, "sleepy", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        beep()  # Generate a beep sound
    else:
        cv2.putText(frame, "active", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Drowsiness Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()


