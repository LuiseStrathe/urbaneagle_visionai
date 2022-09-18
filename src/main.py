import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import webbrowser
st.set_page_config(layout="wide")


## ---------------------INTRO------------------ ##


st.title("Urban Eagle")
st.header("A computer vision project")
st.write("This Project analysis aerial images of urban areas in regards \n"
          "of environmental metrics. As a proof of concept for many \n"
          "metrics, this project showcases two different analytical \n"
          "types based on computer vision machine learning.")
st.write("**Vision**")
st.write("*The sustainability and climate impact of a city* \n"
          "*can be described through many different indicators*.\n"
          "*Up to some extend, these indicators can be exposed by* \n"
          "*analyzing aerial images, including changes in these*\n"
          "*images over time. As of today, there is no easy and*\n"
          "*persistent way to access this information for general* \n"
          "*public or busy decision makers - at least neither quick* \n"
          "*nor unexpensive. Thus, there is a lack of accessibility*\n"
          "*and comparability of these information. Machine learning*\n"
          "*can help to solve this on a scalable and persisting level.*")


## -----------------SIDEBAR---------------------- ##

st.sidebar.title("ABOUT")

repo = "https://github.com/LuiseStrathe/urbaneagle_visionai.git"
if st.sidebar.button("👉  Check out the Urban Eagle repository on GitHub  👈"):
  webbrowser.open_new_tab(repo)

linkedin = "https://www.linkedin.com/in/luise-strathe"
if st.sidebar.button("Find me on linkedIn"):
  webbrowser.open_new_tab(linkedin)

st.sidebar.write("Urban Eagle is a private project, open and free to use.")
st.sidebar.write("New releases with new features will come soon.")

## -----------------Load & SHOW---------------------- ##

st.header("Choose region to show results for!")
image_name = st.selectbox(
  "City", ("Berlin_01", "Berlin_02",
           "Caracas_01",
           "Dresden_01", "Dresden_02", "Dresden_03", "Dresden_04", "Dresden_05", "Dresden_06",
           "Hanoi_01",
           "Marrakesh_01",
           "Potsdam_01", "Potsdam_02",
           #"Rio_01",
           #"South Afrika_01"
           ))

img_original = Image.open(f'data/raw/{image_name}.jpg')
img_trees = Image.open(f'reports/{image_name}/image_trees_bold.jpg')
img_segments = Image.open(f'reports/{image_name}/prediction.png')



## ----------------------RESULTS------------------------ ##

st.text("")
st.text("")
st.write(f'The image covers about 750 by 1200 meters.\n')
st.write(f'A resolution of 25 to 35 cm per pixel is required to perform the estimates.\n')

st.image(img_original, caption=f'The original image {image_name}')

st.text("")
st.text("")
st.header(f"Urban Eagle prediction on {image_name}")
tab1, tab2 = st.tabs(["Semantic Segmentation", "Tree Detection"])


###########################################################
# SEGMENTATION

tab1.subheader("Semantic Segmentation of the surface:")

col1, col2 = tab1.columns(2)
with col1:
  st.image(img_segments, caption=f'Semantic segmentation result', use_column_width=True)
with col2:
  st.write("The surface is segmented into the four follwoing categories represented by colors:")
  st.write(f"\n\nVegetation (GREEN)")
  st.write(f"Water (BLUE)")
  st.write(f"Buildings (RED)")
  st.write(f"Else (sealed surfaces, construction area, train tracks etc.) (BLACK)")
  st.text("The categories are encoded into color channels. \n"
          "Besides RGB, black is shown for RGB==(0, 0, 0), \n"
          "yet in training added as fourth channel for one-hot-encoding.")

#st.metric(label=("vegetation", ""), value=("xx %", "xx %"))


###########################################################
# TREES

tab2.subheader("Identification and localization of trees:")

tab2.warning("Currently **the treshold is always set to 90%**. \n"
              "The selected threshold does not have impact on the results displayed here.")

col3, col4 = tab2.columns(2)

# threshold
col4.write("The threshold can be set by the user. The higher the threshold, the more trees are identified and localized.")
threshold = col4.slider("Select a threshold [%]:", min_value=0, max_value=100, value=90, step=5)
col4.write(f"Selected threshold: **{threshold}%**")
col4.session_state.threshold = threshold

with col3:
  st.write("The trees are identified and localized by a thresholding algorithm.")
  st.write(f"The threshold is set to {threshold}, which means that all image areas \n" 
           f"(25x25 px) with a probability of being a tree higher than {threshold} \n" 
           f"are considered to be a tree.")

tab2.image(img_trees, caption=f'The image with identified trees for {image_name}')

# zoomed in
x1, x2 = np.random.randint(0, 4000, 2)
y1, y2 = np.random.randint(0, 2000, 2)
box1 = (x1, y1, x1+800, y1+400)
box2 = (x2, y2, x2+800, y2+400)
img_original_small = img_original.crop(box1)
img_trees_small = img_trees.crop(box1)

col1, col2 = tab2.columns(2)
with col1:
  st.image(img_trees.crop(box1), use_column_width=True)
with col2:
  st.image(img_trees.crop(box2), use_column_width=True)




###########################################################

st.info('Currently predictions are loaded from file.\n'
        'Live prediction on uploaded images is not yet available.')



## -----------------BACKUP---------------------- ##

# load data
#@st.cache
#def load_data():
#    return to_cache

### st.button("Run me!")






