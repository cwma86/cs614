## Assignment 2 recomender syster

## Colaborative filtering via Matrix factorization

### Details
This repo container extracted code from the mauer-stocks website reccomender 
system to be used for reference realted to my cs614 reccomender system assignment.

### install python dependancies
create a python virtual environement to install the needed dependancies
```bash
pip install venv
python -m venv venv
source ./venv/bin/activate
pip -r install requirement.txt
```

### Running 
This repo contains a snap shot of the website favorites data base, as well as some pre-trained weight matrices. 

To see user reccomendations using the pre-trained weights run one of the follow

`./colab_filter.py -u tech`

`./colab_filter.py -u tech2`

`./colab_filter.py -u finance`

`./colab_filter.py -u finance2`

`./colab_filter.py -u techfinance`

`./colab_filter.py -u defense`

or any other user identified in the sqlite db `favorites.db`

To re-train the model 
`./colab_filter.py --train -u tech`
This will re-factorize the matrix based on favorites stored to the sqlite db `favorites.db`
