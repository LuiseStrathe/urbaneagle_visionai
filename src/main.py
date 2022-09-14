import streamlit as st

import numpy as np
import pandas as pd
import time
from PIL import Image
import os
import webbrowser
st.set_page_config(layout="wide")

## ---------------------HEAD + SIDEBAR------------- ##

st.title("Urban Eagle")
st.write("Applied computer vision for sustainability tracking in urban areas.")

st.sidebar.title("Choose an aerial image!")
image_name = st.sidebar.selectbox(
  "City", ("Berlin_01", "Berlin_02",
           "Caracas_01",
           "Dresden_01", "Dresden_02", "Dresden_03", "Dresden_04", "Dresden_05", "Dresden_06",
           "Hanoi_01",
           "Marrakesh_01",
           "Potsdam_01", "Potsdam_02",
           "Rio_01",
           "South Afrika_01"))
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

## -----------------Load & SHOW---------------------- ##

img_original = Image.open(f'data/raw/images_all/{image_name}.jpg')
img_trees = Image.open(f'reports/{image_name}/image_trees_bold.jpg')
img_segments = Image.open(f'reports/{image_name}/prediction.png')



## ----------------------MAIN------------------------ ##

st.text("")
st.text("")
st.header(f"Urban Eagle prediction on {image_name}!")

st.write(f'The image covers about 750 by 1200 meters.\n')
st.write(f'A resolution of 25 to 35 cm per pixel is required to perform the estimates.\n')

st.image(img_original, caption=f'The original image {image_name}')

st.info('Currently predictions are loaded from file.\n'
        'Live prediction on uploaded images is not yet available.')
# TREES
st.subheader("Identification of trees:")
st.image(img_trees, caption=f'The image with identified trees')


r1, r2 = np.random.randint(0, 1000, 2)
box1 = (r1, r1, r1+400, r1+400)
box2 = (r2, r2, r2+400, r2+400)
img_original_small = img_original.crop(box1)
img_trees_small = img_trees.crop(box1)

col1, col2, col3, col4 = st.columns(4)
with col1:
  st.image(img_original.crop(box1), use_column_width=True)
with col2:
  st.image(img_trees.crop(box1), use_column_width=True)
with col3:
  st.image(img_original.crop(box2), use_column_width=True)
with col4:
  st.image(img_trees.crop(box2), use_column_width=True)

# info threshold

# SEGMENTATION
st.subheader("Semantic Segmentation of the surface:")
st.text(
  f"\n\nGreen: vegetation\n"
  f"Blue: water\n"
  f"Red: buildings\n"
  f"Black: else (sealed surfaces, construction area, train tracks etc.)")

col1, col2 = st.columns(2)
with col1:
  st.image(img_segments, caption=f'Semantic segmentation result', use_column_width=True)
with col2:
  st.image(img_original, caption=f'Original image', use_column_width=True)


#st.metric(label=("vegetation", ""), value=("xx %", "xx %"))


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




