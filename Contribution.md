# Image Colorization

We decided to build and train our model on the Google Colab platform since it provides free GPU access which was critical for our extensive model architecture. 

## Dataset : [Google Landmark](https://s3.amazonaws.com/google-landmark/train/images_001.tar)

### Problems

* We originally set out to build our model with PyTorch and were even getting modest results with it, however it was difficult to find a deployment approach that worked for our specific requirements.
* We explored converting our PyTorch model to JS through the ONNX platform, however the latest version of ONNX.js did not support the upsampling function which was incredibly important for our model. 

### Solutions

* We finally decided to re-write our model in Tensorflow, and surprisingly got much better results than we did with PyTorch.
* For deployment, we decided to go with a simple Streamlit script to have an image uploaded by the user, colorize it and display it back. 
* Streamlit has been offering a new beta service for deploying Streamlit apps through their own platform. Therefore, we only needed to rely on Streamlit to host our webapp. Currently this deployment service is in its beta testing phase and was acquired by us through an invite system.
