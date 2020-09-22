# Image Recolorization

when dealing with images training on a regular PC can take a lot of time. Collab provides free GPUs which would aid us in training the model.

## dataset : Places2

### Problems

* Collab session resets. The dataset needs to be downloaded everytime we need to train the model.
* validation set is the smallest availible about 500mb took 20 mins to download on collab.
* We CAN NOT work together. collab is great for one person but working together on the same notebook is not possible. Collab has some git features but they are not helpful.

### Solutions

We upload 5 - 10 small images on github. We use them as our reference to write and test our code. When we want to properly train our model we will use collab and import all our code as a module. Instructions below.

### Instructions

#### Run on Collab

* get the images
* > wget !wget http://data.csail.mit.edu/places/places365/val_256.tar
* extract the images
* > !tar -xvf val_256.tar
* Git clone |repo ka link|
* >from main import main


#### Run on PC

* git clone |repo ka link|
* python Main.py ~ this will show 1st image from our dataset

* Check the indexing in main file to see whatever image

* change how data is imported at data_things.py
