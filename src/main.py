import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image

## ---------------------HEAD----------------------- ##


st.title("Urban Eagle")
st.write("Vision AI for sustainability tracking in urban areas.")



## -------------------COMMANDS--------------------- ##

left_column, right_column = st.columns(2)
with right_column:
  chosen = st.radio(
    'What do you want to see?',
    ("trees", "area types"))
  st.write(f"You will see  {chosen} for the selected are!")
left_column.radio("Choose a region", ("Dresden_01", "Potsdam_01"))

st.button("Run me!")

## ---------------PREDICTION------------------------ ##

# load data
@st.cache
def load_data():
    return to_cache

data_load_state = st.text('Loading data...')

#st.image('/home/luise/Documents/DataScience/Projects/UrbanEagle/urbaneagle_visionai/data/raw/images_all/Dresden_01.png')
img_original = Image.open('Dresden_01.jpg')
st.image(img_original)



'Starting prediction...'
# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(20):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)
'...and now we\'re done!'





### tree prediction

x = st.slider('x')  # 👈 this is a widget
st.write('Selected threshold:', x, "%")
st.session_state.threshold = x



## ---------------REPORT-------------------------- ##


dataframe = pd.DataFrame(
    np.random.randn(3, 5),
    columns=('col %d' % i for i in range(5)))
st.table(dataframe)

chart_data = pd.DataFrame(
     np.random.randn(5, 1),
     columns=['Number of trees'])

st.line_chart(chart_data)

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)


st.balloons()

