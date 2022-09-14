UrbanEagle_VisionAI
==============================

Let us open the learning eagle eye over cities to the public!


about
-----------------------------------------
computer vision using GoogleEarth aerial images

models: 
1. Object detection of trees (models: ide + pos)
2. Semantic segmentation of surfaces (model: seg)

If you wish to use the model, please contact me via 
<a target="_blank" href="https://www.linkedin.com/in/luise-strathe/">linkedin</a>.
This model was initially developed during my participation at DSR (DataScienceRetreat Berlin) in Batch30.


deployment
-----------------------------------------
current restrictions
- no live prediction
- image resolution ~ 30 cm per pixel
- no uploads or downloads implemented
- object detection reliant on threshold setting

**web app**
[![Streamlit App]\
   (https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]
   (https://share.streamlit.io/luisestrathe/urbaneagle_visionai/main/src/main.py)


Project Organization and folder structure
-----------------------------------------
(not up to date)

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Model training
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
