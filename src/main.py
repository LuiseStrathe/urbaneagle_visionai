import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image
import os
import webbrowser

## ---------------------HEAD + SIDEBAR------------- ##

st.title("Urban Eagle")
st.text("Vision AI for sustainability tracking in urban areas.")

st.sidebar.title("Choose an aerial image!")
image_name = st.sidebar.selectbox(
  "City", ("Berlin_01", "Berlin_02",
           "Caracas_01",
           "Dresden_01", "Dresden_02", "Dresden_03", "Dresden_04", "Dresden_05", "Dresden_06",
           "Hanoi_01",
           "Marakesh_01",
           "Potsdam_01", "Potsdam_02"))
year = st.sidebar.selectbox(
  "Year", ("2021",))

st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("--")
repo = "https://github.com/LuiseStrathe/urbaneagle_visionai.git"
if st.sidebar.button("Check out the urban eagle repository on GitHub 👈"):
  webbrowser.open_new_tab(repo)
st.sidebar.text("Urban Eagle is a private project, open and free to use.")
st.sidebar.text("New releases with new features will come soon.")

## -------------------FUNCTIONS--------------------- ##



## -----------------Load & SHOW---------------------- ##
img_original = Image.open(f'data/raw/images_all/{image_name}.jpg')
img_trees = Image.open(f'reports/{image_name}/image_trees_bold.jpg')
img_segments = Image.open(f'reports/{image_name}/prediction.png')

## ----------------------MAIN------------------------ ##
st.text("")
st.text("")
st.header(f"Urban Eagle prediction on: {image_name}")
st.write(f'The image covers about 750 by 1200 meters.\n')
st.write(f'A resolution of 25 to 35 cm per pixel is required to perform the estimates.\n')

st.image(img_original, caption=f'The original image')

st.info('Currently predictions are loaded from file.\n'
        'Live prediction on uploaded images is not yet available.')
# TREES
st.subheader("Find trees:")
st.image(img_trees, caption=f'The image with identified trees')
# info threshold

# SEGMENTATION
st.subheader("Evaluate surface:")
left_column, right_column = st.columns(2)
left_column.image(img_segments, caption=f'The image with areas identified')
right_column.text(
  f"\n\nGreen: vegetation\n"
  f"Blue: water\n"
  f"Red: buildings\n"
  f"Black: else (sealed surfaces, construction area, train tracks etc.)")




## -----------------BACKUP---------------------- ##

# load data
#@st.cache
#def load_data():
#    return to_cache

### st.button("Run me!")

### slider

#x = st.slider('x')  # 👈 this is a widget
#st.write('Selected threshold:', x, "%")
#st.session_state.threshold = x




