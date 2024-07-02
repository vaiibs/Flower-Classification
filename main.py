import tensorflow as tf
from tensorflow import keras
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

model = load_model('Flower_Recog_Model.keras')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

img_size = 180


def classify_image(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(img_size, img_size))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    outcome = 'The Image belongs to {} with accuracy of {:0.2f}'.format(flower_names[np.argmax(predictions)],
                                                                         np.max(predictions) * 100)
    return outcome


st.header('Flower Classification CNN Model')
uploaded_file = st.file_uploader('Upload an Image to classify :')
if uploaded_file is not None:
    st.image(uploaded_file, width = 250)
    st.write(classify_image(uploaded_file))
