# Import necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter import Tk, filedialog
from PIL import Image
import io

# Function to apply CLAHE to an image
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    # Apply CLAHE to the grayscale image
    enhanced_image = clahe.apply(gray_image)
    return enhanced_image

# Function to resize the image to 416x416 pixels
def resize_image(image, size=(416, 416)):
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized_image

# Function to upload an image from the computer
def upload_image():
    # Create a Tkinter root window
    root = Tk()
    root.withdraw()  # Hide the root window
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if file_path:
        # Read the selected image
        image = cv2.imread(file_path)
        return image
    else:
        return None

# Function to display images using matplotlib
def display_images(original_image, enhanced_image):
    # Convert images from BGR to RGB for displaying
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
    
    # Create a figure with two subplots
    plt.figure(figsize=(12, 6))
    
    # Display original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image_rgb)
    plt.axis('off')
    
    # Display enhanced image
    plt.subplot(1, 2, 2)
    plt.title("Enhanced Image")
    plt.imshow(enhanced_image_rgb, cmap='gray')
    plt.axis('off')
    
    # Show the figure
    plt.show()

# Main function to execute the workflow
def main():
    # Upload an image
    original_image = upload_image()
    if original_image is not None:
        # Apply CLAHE to the image
        enhanced_image = apply_clahe(original_image)
        # Resize the enhanced image
        resized_image = resize_image(enhanced_image)
        # Display the original and enhanced images
        display_images(original_image, resized_image)
    else:
        print("No image selected. Please try again.")

# Run the main function
if __name__ == "__main__":
    main()
