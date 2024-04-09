#You can find that in Hugging Face
import streamlit as st
import time
import uuid 
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image 
st.title("Traffic signs application :camera:")
img=st.camera_input("Camera")
model=load_model("my_model.h5")
def process_image(input_img):
    input_img=input_img.resize((170,170)) 
    input_img=np.array(input_img)
    input_img=input_img/255.0
    input_img=np.expand_dims(input_img,axis=0)
    return input_img
if img is not None:
    img=Image.open(img)
    st.image(img,caption="Uploaded Image")
    image=process_image(img)
    prediction=model.predict(image)
    predicted_class=np.argmax(prediction)    
    class_names=[
    'Speed limit (20km/h)',
    'Speed limit (30km/h)',
    'Speed limit (50km/h)',
    'Speed limit (60km/h)',
    'Speed limit (70km/h)',
    'Speed limit (80km/h)',
    'End of speed limit (80km/h)',
    'Speed limit (100km/h)',
    'Speed limit (120km/h)',
    'No passing',
    'No passing veh over 3.5 tons',
    'Right-of-way at intersection',
    'Priority road',
    'Yield',
    'Stop',
    'No vehicles',
    'Veh > 3.5 tons prohibited',
    'No entry',
    'General caution',
    'Dangerous curve left',
    'Dangerous curve right',
    'Double curve',
    'Bumpy road',
    'Slippery road',
    'Road narrows on the right',
    'Road work',
    'Traffic signals',
    'Pedestrians',
    'Children crossing',
    'Bicycles crossing',
    'Beware of ice/snow',
    'Wild animals crossing',
    'End speed + passing limits',
    'Turn right ahead',
    'Turn left ahead',
    'Ahead only',
    'Go straight or right',
    'Go straight or left',
    'Keep right',
    'Keep left',
    'Roundabout mandatory',
    'End of no passing',
    'End no passing veh > 3.5 tons']
    st.write(class_names[predicted_class])
    st.link_button("Find Jupyter Notebook File In My GitHub Account",url="https://github.com/emirkaanozdemr")
