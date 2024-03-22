from skimage import io, color, measure
from skimage.measure import find_contours
from math import sqrt
# from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os

def get_sorted_images(folder_path):
    # Collect all jpg images in the folder
    images = glob.glob(os.path.join(folder_path, '*.jpg'))
    
    # Sort images based on the numerical value in their names
    images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace('blue3ten', '')))
    return images

def read_image(image_path):
    return io.imread(image_path)

def get_cropping_mask(image_shape, point1, point2):
    # Calculate the slope (m) and y-intercept (b) of the line
    m = (point2[1] - point1[1]) / (point2[0] - point1[0])
    b = point1[1] - m * point1[0]
    
    # Generate a grid of coordinates corresponding to the indices of the image
    y_indices, x_indices = np.indices(image_shape)
    
    # Calculate the x-coordinates on the line corresponding to each y-coordinate
    x_line = (y_indices - b) / m
    
    # Create a mask for pixels below the line
    mask = x_indices > x_line
    
    return mask

def crop_image(image, mask):
    # Copy the image to not alter the original
    cropped_image = image.copy()
    
    # Set pixels above the line to white (or any other background color)
    cropped_image[~mask] = 255  # Assuming the image is in RGB or grayscale
    return cropped_image

def convert_to_hsv(image):
    return color.rgb2hsv(image)

def create_binary_mask(image_hsv, saturation_range, value_range):
    return (
        (image_hsv[:, :, 1] >= saturation_range[0]) & (image_hsv[:, :, 1] <= saturation_range[1]) &
        (image_hsv[:, :, 2] >= value_range[0]) & (image_hsv[:, :, 2] <= value_range[1])
    )

def measure_average_radius(image_binary_mask, pixels_per_cm):
    try:
        # Label the image
        label_img = measure.label(image_binary_mask)
        regions = measure.regionprops(label_img)

        # Check if any regions were detected
        if not regions:
            return np.nan

        drop_region = max(regions, key=lambda r: r.area)
        centroid = drop_region.centroid

        contours = find_contours(image_binary_mask, 0.5)
        if not contours:
            return np.nan

        largest_contour = max(contours, key=lambda x: x.shape[0])
        distances = np.sqrt((largest_contour[:, 0] - centroid[0]) ** 2 + (largest_contour[:, 1] - centroid[1]) ** 2)
        average_radius_pix = np.mean(distances)
        average_radius = average_radius_pix / pixels_per_cm

        return average_radius
    except Exception as e:
        print(f"Error processing image: {e}")
        return np.nan

def pixel_size(ruler):
    length, width = ruler.shape[:2]
    distance = sqrt(length**2 + width**2)
    pixels_per_cm = distance / 5
    return pixels_per_cm

saturation_range = [0.5, 1]  # Minimum saturation of the drop
value_range = [0.05, 0.84]  # Maximum value of the drop value

ruler_path = 'image_data/blue3tenruler.jpeg'
ruler = read_image(ruler_path)
pixels_per_cm = pixel_size(ruler)

csv_file_path = os.path.join('image_data', 'extracted_blue_data.csv')

if os.path.exists(csv_file_path):
    df = pd.read_csv(csv_file_path)
    processed_paths = df['Image_Path'].tolist()
else:
    df = pd.DataFrame(columns=['Image_Path', 'Average_Radius'])
    processed_paths = []
    
image_path = 'image_data/blue_spot/blue3ten1100.jpg'

initial_image = read_image(image_path)  # Use the first image to get the dimensions
mask = get_cropping_mask(initial_image.shape[:2], (978, 881), (2861, 856))

image = read_image(image_path)
cropped_image = crop_image(image, mask)
image_hsv = convert_to_hsv(cropped_image)
image_binary_mask = create_binary_mask(image_hsv, saturation_range, value_range)
average_radius = measure_average_radius(image_binary_mask, pixels_per_cm)

print("Average radius:", average_radius)
# Plot the images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Adjust the figure size as needed

# Original Image
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Binary Mask
# Displaying the binary mask in grayscale
axes[1].imshow(image_binary_mask, cmap='gray')
axes[1].set_title('Binary Mask')
axes[1].axis('off')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

