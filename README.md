UrbanEagle_VisionAI
==============================
<br>
<b>Let's open the learning eagle eye over cities for everyone who can make good use of it!</b>

<br>

***MOTIVATION***

*The sustainability and climate impact of a city can be described through many different indicators. Up to some extend, these indicators can be exposed by analyzing aerial images, including changes in these images over time. As of today, there is no easy and persistent way to access this information for general public or busy decision makers - at least neither quick nor unexpensive. Thus, there is a lack of accessibility and comparability of these information. Machine learning can help to solve this on a scalable and persisting level.*

<br>

**ABSTRACT**

This Project analysis aerial images of urban areas in regards of environmental metrics. As a proof of concept for many metrics, this project showcases two different analytical types based on computer vision machine learning.

First - **the identification and counting of many small, similar objects (trees)**

Second - **the semantic segmentation of the surface into relevant groups**

<br>

> ℹ️ <a target="_blank" href="https://share.streamlit.io/luisestrathe/urbaneagle_visionai/main/src/main.py"> Visit the deployed STREAMLIT web app here! </a>
   


> ℹ️ <a target="_blank" href="https://github.com/LuiseStrathe/urbaneagle_visionai/blob/main/UrbanEagle_final.pdf"> Check out the project presentation </a>

<br>


**ABOUT**

X Luise Strathe

This model was initially developed during my participation during my data science bootcamp at <a target="_blank" href="https://www.linkedin.com/in/luise-strathe/">Data Science Retreat Berlin</a> in Batch 30.

If you wish to use the model, please contact me via 
<a target="_blank" href="https://www.linkedin.com/in/luise-strathe/">linkedin</a>.
<br><br> 

<br><br>

----
<br>

APPROACH
----------------------------------------

<br>


1) INPUT  
- Use aerial images of a city (or urban area) --> GoogleEarth
- Source: GoogleEarth (open)
- Dimensions (restricted by GE): 2.400 x 4.800 px (equals ~ 750 x 1.250 m)
- GoogleEarth allows historic images and a very high global coverage

<br>

2) PROCESSING / ML MODELS

    **1a.   Object detection** of trees by CNN with logistic regression per image tile ("ide")

    **1b.   Fine-positioning** of detected trees in a small area with regression model ("pos)

    **2.    Semantic segmentation** of surfaces ("seg")

<br>

3) OUTCOME

- <b>TREE DETECTION:</b> Identification, count, localization of trees in the area
- <b>SEMANTIC SEGMENTATION:</b> Surface is separated into four surface types
   (green: vegetation, red: buildings, blue: water, black: enclosed/unused area)
- Visual representation for both applications
- Results for example images are deployed via STEAMLIT (static)

<br>

4) USE CASES
- Information about sustainability indicators of an urban area
- Quantifiable comparison of area metrics over time
- Quantifiable comparison of different regions (e.g. cities or districts) 

<br>
<b>The applied models showcase the feasibility of using image segmentation and multi-object detection. This can be introduced in similar manner to many more metrics.

Furthermore images of large areas can (e.g. full cities) may be concatenated.</b>


<br>



**RESTRICTIONS & OUTLOOK**

> ℹ️ The result of the portfolio project is an proof of concept. Potential to grow: e.g. implement more cities, timestamps and metrics or improve the user interface and user options (filters, visualizations).
If you are in an NGO or similar and have interest in profiting from this project, lease contact me.
< 


CURRENT RESTRICTIONS:
- no live prediction
- image resolution ~ 30 cm per pixel (free satellite images not available)
- no uploads or downloads implemented
- object detection reliant on threshold setting
- geo-coordinates not used

REPORTING
- Allow creation of reports for an uploaded image
- Allow Comparison for diferrent images
- Allow concatenation of images --> cover larger areas
- Integration of image generation directly via UrbanEagle

POTENTIAL ADDITIONAL METRICS:
- Detect & count trees in the area of the city
optional metrics
- Display new and gone trees
- Categorize trees (by size/ health)
- Count cars or other objects large enough
- Show green-ratio (natural area, water area, enclosed area segmentation)
- Identify how well vegetation is doing (eventually include annual precipitation/ temperature)
- Identify sustainable energy elements (solar panels) or free spaces for them
- Identify available space for green rooftops

POTENTIAL TARGET USERS:
- Sustainability-conscious citizens and local initiatives
- NGO's with focus on urban sustainability and development
- Political parties and local decision makers




<br><br> 

----
<br>

REPO STRUCTURE
-----------------------------------------

    
    |── data
    │   ├── input          <- Images and other data as inputs for processing
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained models and according training reports
    │
    ├── notebooks          <- Jupyter notebooks for exploration and model training
    |                         Includes training of positioning & segmentation
    │
    ├── reports            <- Generated analysis and images
    │
    ├── requirements.txt   
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can 
    |                         be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── data           <- Scripts to download or generate data
    │   ├── helper         
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions
    │   |── visualization  <- Scripts to visualize images and results
    |   |
    |   |── identification_train        <- Script to train identification model
    |   |── main                        <- Script to deploy project via STREAMLIT
    |   |── predict_segments            <- Script segmentation of an aerial image
    │   └── predict trees               <- Script predict trees for an aerial image
    |
    |── tox.ini            <- tox file with settings for running tox; 
    |                         see tox.readthedocs.io
    ├── LICENSE 
    ├── Makefile    
    └── README   





<br><br> 

----
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
