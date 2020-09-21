# Image Recolorization

when dealing with images training on a regular PC can take a lot of time. Collab provides free GPUs which would aid us in training the model.

### dataset : Places2

##### Problem
* Collab session resets. The dataset needs to be downloaded everytime we need to train the model.
* validation set is the smallest availible about 500mb took 20 mins to download on collab.
* We CAN NOT work together. collab is great for one person but working together on the same notebook is not possible. Collab has some git features but they are not helpful.

##### Solution
We upload 5 - 10 small images on github. We use them as our reference to write and test our code. When we want to properly train our model we will use collab and import all our code as a module. Instructions below.

##### Instructions
