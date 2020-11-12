import streamlit as st
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

def load_model(path):
	model = keras.models.load_model(path)
	model.summary()
	return model

def preProc(A):
	A = np.array(A)
	normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
	B = normalization_layer(A)
	B = tf.image.resize(B, [1024, 1024])
	A = tf.image.rgb_to_hsv(B)
	return A[:,:,:,-1:]

def recolor(model, img_array):
	
	print(img_array.shape)
	predImg = model.predict(img_array, verbose=0)
	predImg = tf.image.hsv_to_rgb(predImg)
	return np.array(predImg)

def main():
	link = 'https://images.unsplash.com/photo-1558056524-97698af21ff8?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1050&q=80'
	st.image(link, use_column_width=True)

	st.title("Supercolor ~ Colorful Image Colorization")

	st.subheader("Supercolor is an ML web-app which has been trained to rebuild color images from their grayscale or B/W input equivalents that you provide us with.")

	st.write("It works on the concept of Variational Autoencoders. This approach was introduced by Richard Zhang in his paper [Colorful Image Colorization](https://arxiv.org/abs/1603.08511). These autoencoders cleverly store the important details of a big image into a small space and then try to recreate this image in color. We penalize the autoencoder when it doesn't do a good job until it begins to get it right.", unsafe_allow_htl=True)
	st.write("Feel free to head to our [Github repository](https://github.com/Data-Science-Community-SRM/Image-Recolorization) to explore the code.")

	st.write("Keep in mind that it may take us some time to colorize this image.")

	img_file_buffer = st.file_uploader("Upload Black & White Image", type=['png', 'jpg'])
	st.set_option('deprecation.showfileUploaderEncoding', False)

	model_path = 'recolor.h5'
	if img_file_buffer is not None:
		
		with st.spinner("Colorizing Image..."):
			model = load_model(model_path)

			image = Image.open(img_file_buffer)
			st.image(image, caption="Original Image", use_column_width=True)
			img_array = np.array(image)

			img = preProc([img_array])
			
			colorImg = recolor(model, img)
			st.image(colorImg, caption = "Colorized Image", use_column_width=True)
		
		st.success("Successfuly colorized the image!")

if __name__ == "__main__":
	main()
