import streamlit as st
import cv2
from PIL import Image
import numpy as np
import mediapipe as mp 
import numpy as np 

# Initializing mediapipe segmentation class.
mp_selfie_segmentation = mp.solutions.selfie_segmentation
# Setup segmentation function.
segment = mp_selfie_segmentation.SelfieSegmentation()

# Title.
st.title("Subtract construction scene and workers")

# Add checkboxes.
mode = st.sidebar.checkbox('Blur', value = False)
ch_bg  = st.sidebar.checkbox('Change Background', value = False)
des_bg = st.sidebar.checkbox('Desaturate Background', value = False)

# Add slider to control kernel size for blurring.
if mode == True:
	ksize = st.sidebar.slider("Kernel Size", 
		min_value = 3, max_value = 45, step = 2, value = 15)

# Segmentation threshold.
threshold = st.sidebar.slider("Threshold", min_value = 0.1, max_value = 1.0, step = 0.01, value = 0.3)

# Function to generate mask.
def generateMask(img):
	image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	results = segment.process(image_rgb)
	binary_mask = results.segmentation_mask > threshold
	mask = np.dstack((binary_mask, binary_mask, binary_mask))
	return mask

# File uploader button for original image.
original_image = st.file_uploader("Choose original image", type =['jpg','jpeg','jfif','png'])

# If change background is selected.
if ch_bg == True:
	# Load file uploader button for background image.
	background_image = st.file_uploader("Choose background image", type = ['jpg','jpeg','jfif','png'])
	if background_image is not None:
		# Convert uploaded image into numpy array.
		bg_image = np.array(Image.open(background_image))

# Placeholders to display images.
placeholders = st.columns(2)
container    = st.columns(1)


if original_image is not None:
	image = np.array(Image.open(original_image))
	placeholders[0].image(image)
	mask = generateMask(image)


try:
	if ch_bg == True:
		placeholders[1].image(bg_image)

	if mode == True:	
		if ch_bg == True:
			if background_image is not None:
				# Convert to numpy array.
				bg_image = np.array(Image.open(background_image))
				# Resize to match dimension with original image.
				bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
				# Blur it.
				bg_blurred = cv2.GaussianBlur(bg_image, (ksize,ksize), 0)
				# Desaturate background?
				if des_bg == True:
					bg_blurred = cv2.cvtColor(bg_blurred, cv2.COLOR_BGR2GRAY)
					bg_blurred = cv2.cvtColor(bg_blurred, cv2.COLOR_GRAY2BGR)
				else:
					pass
				# Modify image.
				output_image = np.where(mask, image, bg_blurred)
				# Display.
				container[0].image(output_image)
		else:
			# Blur the image.
			img_blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
			# Desaturate background?
			if des_bg == True:
				img_blurred = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
				img_blurred = cv2.cvtColor(img_blurred, cv2.COLOR_GRAY2BGR)
			else:
				pass
			# Modify image.
			output_image = np.where(mask, image, img_blurred)
			# Display.
			container[0].image(output_image)

	else:
		if ch_bg == True:
			if background_image is not None:
				# Convert to numpy array.
				bg_image = np.array(Image.open(background_image))
				# Resize to match dimension with original image.
				bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
				# Desaturate background?
				if des_bg == True:
					bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2GRAY)
					bg_image = cv2.cvtColor(bg_image, cv2.COLOR_GRAY2BGR)
				else:
					pass
				# Modify image.
				output_image = np.where(mask, image, bg_image)
				# Display.
				container[0].image(output_image)
		else:
			# Desaturate background?
			if des_bg == True:
				img_des = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				img_des = cv2.cvtColor(img_des, cv2.COLOR_GRAY2BGR)
				# Modify image.
				output_image = np.where(mask, image, img_des)
			else:
				output_image = image
			# Display.
			container[0].image(output_image)
except:
	pass
	