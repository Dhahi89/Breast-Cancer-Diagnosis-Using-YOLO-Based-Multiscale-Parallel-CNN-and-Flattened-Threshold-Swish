# Import necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter import Tk, filedialog

# Function to rotate an image by a specified angle
def rotate_image(image, angle):
    # Get the dimensions of the image
    (h, w) = image.shape[:2]
    # Calculate the center of the image
    center = (w // 2, h // 2)
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image

# Function to inject Gaussian noise into an image
def inject_gaussian_noise(image, mean=0, stddev=25):
    # Generate Gaussian noise
    gaussian_noise = np.random.normal(mean, stddev, image.shape).astype('uint8')
    # Add the Gaussian noise to the image
    noisy_image = cv2.add(image, gaussian_noise)
    return noisy_image

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
def display_images(images, titles):
    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 8))
    num_images = len(images)
    
    for i in range(num_images):
        # Convert image from BGR to RGB for displaying
        image_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        plt.subplot(1, num_images, i + 1)
        plt.title(titles[i])
        plt.imshow(image_rgb)
        plt.axis('off')
    
    # Show the figure
    plt.show()

# Main function to execute the workflow
def main():
    # Upload an image
    original_image = upload_image()
    if original_image is not None:
        # Perform rotations
        rotated_90 = rotate_image(original_image, 90)
        rotated_180 = rotate_image(original_image, 180)
        rotated_270 = rotate_image(original_image, 270)
        
        # Inject Gaussian noise
        noisy_image = inject_gaussian_noise(original_image)
        
        # Display the original and augmented images
        images = [original_image, rotated_90, rotated_180, rotated_270, noisy_image]
        titles = ["Original Image", "Rotated 90°", "Rotated 180°", "Rotated 270°", "Noisy Image"]
        display_images(images, titles)
    else:
        print("No image selected. Please try again.")

# Run the main function
if __name__ == "__main__":
    main()
