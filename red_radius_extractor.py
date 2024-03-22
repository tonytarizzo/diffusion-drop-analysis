from skimage import io, color, measure
from skimage.measure import find_contours
from math import sqrt
from tqdm import tqdm
import numpy as np
import pandas as pd
import glob
import os

def get_sorted_images(folder_path):
    # Collect all jpg images in the folder
    images = glob.glob(os.path.join(folder_path, '*.jpg'))
    
    # Sort images based on the numerical value in their names
    images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace('redten', '')))
    return images

def read_image(image_path):
    return io.imread(image_path)

def convert_to_hsv(image):
    return color.rgb2hsv(image)

def create_binary_mask(image_hsv, hue_range, saturation_range):
    return (
        (image_hsv[:, :, 0] >= hue_range[0]) & (image_hsv[:, :, 0] <= hue_range[1]) &
        (image_hsv[:, :, 1] >= saturation_range[0]) & (image_hsv[:, :, 1] <= saturation_range[1])
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

hue_range = [0.77, 0.9]  # Pink hue range
saturation_range = [0.5, 1]  # Minimum saturation of the pink color

ruler_path = 'image_data/redtenruler.jpg'
ruler = read_image(ruler_path)
pixels_per_cm = pixel_size(ruler)

csv_file_path = os.path.join('image_data', 'extracted_red_data.csv')

if os.path.exists(csv_file_path):
    df = pd.read_csv(csv_file_path)
    processed_paths = df['Image_Path'].tolist()
else:
    df = pd.DataFrame(columns=['Image_Path', 'Average_Radius'])
    processed_paths = []

folder_path = 'image_data/red_spot/'
image_paths = get_sorted_images(folder_path)

data_to_append = []

for path in tqdm(image_paths, desc="Processing images"):
    if path not in processed_paths:
        try:
            image = read_image(path)
            image_hsv = convert_to_hsv(image)
            image_binary_mask = create_binary_mask(image_hsv, hue_range, saturation_range)
            average_radius = measure_average_radius(image_binary_mask, pixels_per_cm)
            if np.isnan(average_radius):
                print(f"No valid drop detected in {path}, logging as NaN.")
        except Exception as e:
            print(f"Unexpected error processing {path}: {e}")
            average_radius = np.nan
    
        data_to_append.append({'Image_Path': path, 'Average_Radius': average_radius})

if not df.empty:
    # If df is not empty, concatenate the new data
    new_df = pd.DataFrame(data_to_append)
    df = pd.concat([df, new_df], ignore_index=True)
else:
    # If df is empty, directly create it from the data collected
    df = pd.DataFrame(data_to_append)

# Save the DataFrame to CSV
df.to_csv(csv_file_path, index=False)