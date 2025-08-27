import cv2 as cv
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobil

model = tf.keras.models.load_model(r"Waste_Classification_CNN.h5")

st.title("Waste Classification CNN")
file_upload = st.file_uploader("Upload an image")



if file_upload is not None:
    st.image(file_upload)
    img = Image.open(file_upload)
    img = img.resize((224,224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img/255
    img = np.expand_dims(img,axis=0)
    result = model.predict(img)
    btn=st.button("Check")
    if btn:
        if (result[0][1]>0.5):
            obj="Recyclable"
            st.write(f'This object is {obj} object')
        else:
            obj='Organic'
            st.write(f'This object is {obj} object')




