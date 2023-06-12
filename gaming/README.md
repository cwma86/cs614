# Gaming assignment
This directory impelments a simple blackjack game and a deepq network trained to play blackjack 

## install
install python3 and pip

install the python venv package

`python3 -m pip install venv`

Create a virtual python environment and activate it 

`python -m venv venv`
`source venv/bin/activate`

Install pip requirements 
`python3 -r requirements.txt`

## Black jack
The `black_jack.py` is a simple single deck blackjack game designed to be used by a user and also following the python package `gym` API to allow for easy training of reinforcement learning models

### Usage
`./blackjack`
You will be presented with the sum of your two cards and the single card the dealer is holding and asked to stay or hit via user input to the terminal

### black jack deep q learning script
This script will use the black jack code to train and evaluate a deep q learning agent
### Usage
`./black_jack_deepq.py`

see the help menu for full list of options

