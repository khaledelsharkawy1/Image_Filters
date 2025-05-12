import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#from tkinter import filedialog

def apply_filter(image, kernel):
    image_array = np.array(image)
    image_height, image_width = image_array.shape[:2]
    kernel_height, kernel_width = kernel.shape
    filtered_image = np.zeros_like(image_array)
    
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image_array, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant')
    
    for i in range(image_height):
        for j in range(image_width):
            roi = padded_image[i:i+kernel_height, j:j+kernel_width]
            for c in range(3):
                filtered_image[i, j, c] = np.sum(roi[:, :, c] * kernel)
    
    return np.clip(filtered_image, 0, 255).astype(np.uint8)

# Gaussian Filter (low-pass filter)
def gaussian_kernel(size, sigma=1.0):
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma**2)) * np.exp(- ((x - (size // 2))**2 + (y - (size // 2))**2) / (2 * sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

# Median Filter
def median_filter(image, kernel_size):
    image_array = np.array(image)
    pad_size = kernel_size // 2
    padded_image = np.pad(image_array, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
    filtered_image = np.zeros_like(image_array)
    
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            roi = padded_image[i:i+kernel_size, j:j+kernel_size]
            for c in range(3):  # For each color channel
                filtered_image[i, j, c] = np.median(roi[:, :, c])
    
    return filtered_image

def laplacian_sharpen(image):
    output = np.copy(image).astype(float)
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    
   
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            neighborhood = image[y-1:y+2, x-1:x+2]
            output[y, x] = np.sum(neighborhood * kernel)
    
   
    sharpened = image + output
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def unsharp_mask(image, amount=1.5):
   
    blurred = np.copy(image).astype(float)
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            neighborhood = image[y-1:y+2, x-1:x+2]
            blurred[y, x] = np.mean(neighborhood)
    
    
    mask = image.astype(float) - blurred
    sharpened = image + amount * mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)


import random

def add_salt_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    height, width = image.shape

    num_salt = np.ceil(salt_prob * image.size)
    for i in range(int(num_salt)):
        x = random.randint(0, height - 1)
        y = random.randint(0, width - 1)
        noisy_image[x, y] = 255

    num_pepper = np.ceil(pepper_prob * image.size)
    for i in range(int(num_pepper)):
        x = random.randint(0, height - 1)
        y = random.randint(0, width - 1)
        noisy_image[x, y] = 0

    return noisy_image


# === Streamlit App ===
import streamlit as st
st.title("Image Filtering App")

uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
filter_option = st.selectbox("Choose a filter", [
    "None", "Gaussian Blur", "Median Filter", "Laplacian Sharpen",
    "Unsharp Mask", "Salt & Pepper Noise"
])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)
    img_np = np.array(image)

    if st.button("Apply Filter"):
        if filter_option == "None":
            result = img_np
        elif filter_option == "Gaussian Blur":
            result = apply_filter(image, gaussian_kernel(3, sigma=1.0))
        elif filter_option == "Median Filter":
            result = median_filter(image, 3)
        elif filter_option == "Laplacian Sharpen":
            result = laplacian_sharpen(img_np)
        elif filter_option == "Unsharp Mask":
            result = unsharp_mask(img_np)
        elif filter_option == "Salt & Pepper Noise":
            gray = np.array(image.convert("L"))
            result = add_salt_pepper_noise(gray, 0.02, 0.02)
        else:
            result = img_np

        # Display Result
        st.image(result, caption=f"Filtered Image ({filter_option})", use_container_width=True)