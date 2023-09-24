import cv2
import numpy as np
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

# Load the pretrained InceptionV3 model for classification
model = InceptionV3(weights='imagenet')

# Open video capture
cap = cv2.VideoCapture('C:\\Users\\Kavya Srivastava\\OneDrive - Graphic Era University\\Desktop\\graymatics\\tensorflow\\tensorflow\\examples\\label_image\\data\\elephant.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for classification
    resized_frame = cv2.resize(frame, (299, 299))
    preprocessed_frame = preprocess_input(np.expand_dims(resized_frame, axis=0))

    # Perform classification
    predictions = model.predict(preprocessed_frame)
    predicted_class = decode_predictions(predictions)[0][0]

    # Display label and confidence on the frame
    label = f"{predicted_class[1]}: {predicted_class[2]*100:.2f}%"
    confidence = predicted_class[2] * 100
    color = (0, 255, 0) if confidence > 80 else (0, 0, 255)
    
    # Display confidence on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Display confidence level in a separate line
    cv2.putText(frame, f"Confidence: {confidence:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the frame
    cv2.imshow('demo', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()