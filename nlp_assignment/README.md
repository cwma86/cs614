# Assignment 3 
# NLP 

## install
build a virtual environement and install the requirements.txt
**Note** assumes python3 and pip installation is already present 

start by cloning this repo and navigating the the directory that contains this Readme
`python3 -m pip install venv`
`python -m venv venv`
`source venv/bin/activate`
`pip install -r requirements.txt`

## pulling data
This script will enable the user to pull new data from the news API and store those articles to the sqlite3 db news.db
`./pull_descriptions`

## labeling_truth
This script will read from the sqlite3 db news.db, find stories that have not yet been labeled, and then request labeling guidance from the user in the terminal
`./label_data`

## training a model
This script pull all labeled data from the database, splits test and train dataset, trains the model, and then provides various evalutation metrics and example predicitions. 
`./test_train_model.py`
