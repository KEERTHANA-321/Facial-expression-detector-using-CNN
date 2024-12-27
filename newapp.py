import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

model = load_model('emotion_model.h5')

def preprocess_image(image):
    if isinstance(image, np.ndarray): 
        img_array = image
    else:
        img_array = np.array(image)
   if len(img_array.shape) == 2:
        gray = img_array
    elif len(img_array.shape) == 3: 
        if img_array.shape[2] == 3: 
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) 
        else:
            raise ValueError("Unexpected image format! Image must be 1 or 3 channels.")
    else:
        raise ValueError("Unexpected image format! Image must be 2D (grayscale) or 3D (RGB/BGR).")
    resized=cv2.resize(gray,(48,48))
    normalized=resized/255.0
    input=np.expand_dims(normalized,axis=0)
    input=np.expand_dims(input,axis=-1)
    return input

#App UI
st.set_page_config(page_title="Facial Expression Detection App",layout="centered")
st.title("Facial Expression Detection App")
st.write("Upload an image or use your webcam to detect emotions in faces. The app predicts one of the following emotions:")
st.markdown("**Angry**, **Disgust**, **Fear**, **Happy**, **Sad**, **Surprise**, **Neutral**")

#Sidebar
st.sidebar.header("Instructions")
st.sidebar.write("1. Upload an image with a visible face or capture using the webcam.\n"
                 "2. The app will display the detected emotion.")
st.subheader("Choose an Option")
option=st.radio("How would you like to provide an image?", ("Upload an Image", "Use Webcam"))
#File Uploader
if option=="Upload an Image":
    st.subheader("Upload an Image")
    uploaded_file=st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
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


elif option == "Use Webcam":
    st.subheader("Use Your Webcam")
    webcam_input=st.camera_input("Capture an image with your webcam")
    if webcam_input is not None:
        webcam=np.array(Image.open(webcam_input))
        st.image(webcam,caption="Captured Image",use_container_width=True)
        #Preprocessing
        input_image=preprocess_image(webcam)
        #prediction
        predictions=model.predict(input_image)
        predicted_class=np.argmax(predictions)
        emotion=emotion_labels[predicted_class]
        st.success(f"**Predicted Emotion: {emotion}**")
