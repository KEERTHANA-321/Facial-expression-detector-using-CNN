import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

model = load_model('emotion_model.h5')

def preprocess_image(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    resized=cv2.resize(gray,(48,48))
    normalized=resized/255.0
    input=np.expand_dims(normalized,axis=0)
    input=np.expand_dims(input,axis=-1)
    return input

#App UI
st.set_page_config(page_title="Facial Expression Detection App",layout="centered")
st.title("Facial Expression App")
st.write("Upload an image or use your webcam to detect emotions in faces. The app predicts one of the following emotions:")
st.markdown("**Angry**, **Disgust**, **Fear**, **Happy**, **Sad**, **Surprise**, **Neutral**")

#Sidebar
st.sidebar.header("Instructions")
st.sidebar.write("1. Upload an image with a visible face or capture using the webcam.\n"
                 "2. The app will display the detected emotion.")
#File Uploader
st.subheader("Upload an Image")
uploaded_file=st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#Webcam
st.subheader("Use Your Webcam")
use_webcam=st.button("Capture")

if uploaded_file is not None:
    image=np.array(Image.open(uploaded_file))
    st.image(image,caption="Uploaded Image", use_container_width=True)
    #Preprocessingg the image
    input_image=preprocess_image(image)
    #prediction
    predictions=model.predict(input_image)
    predicted_class=np.argmax(predictions)
    emotion=emotion_labels[predicted_class]
  
    st.success(f"**Predicted Emotion: {emotion}**")
elif use_webcam:
    st.write("Opening your webcam...")
    capture=cv2.VideoCapture(0)
    
    if capture.isOpened():
        ret, frame = capture.read()
        if ret:
            st.image(frame, caption="Captured Frame",use_container_width=True)
            #Preprocessing
            input_image=preprocess_image(frame)
            #prediction
            predictions=model.predict(input_image)
            predicted_class=np.argmax(predictions)
            emotion=emotion_labels[predicted_class]
            st.success(f"**Predicted Emotion: {emotion}**")
        else:
            st.error("Unable to capture an image from the webcam.")
        capture.release()
    else:
        st.error("Could not open the webcam. Ensure it's properly connected.")
