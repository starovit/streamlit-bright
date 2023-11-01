import matplotlib.pyplot as plt
import cv2
import json
import numpy as np

plt.rcParams.update({'font.size': 18})


# save utils
def create_figure(image: np.ndarray,
                areas_mask: dict,
                areas_histogram: dict,
                areas_center: dict):
    """
    Create a visualization for different areas of a bright image, add contours, and text annotations,
    and save it to the specified output path.

    Parameters:
        bright_image (numpy.ndarray): The bright image as a NumPy array (grayscale).
        area_mask (dict): Dictionary containing area names as keys and their respective masks as values.
        area_histogram (dict): Dictionary containing area names as keys and their respective histograms as values.
        area_center (dict): Dictionary containing area names as keys and their respective centroid coordinates as values.
        output_path (str): The file path where the image will be saved.

    Example:
        image = np.array([...])  # Grayscale or RGB image as a NumPy array
        area_mask = {'area1': np.array([...]), 'area2': np.array([...]), ...}  # Masks for each area
        area_histogram = {'area1': np.array([...]), 'area2': np.array([...]), ...}  # Histograms for each area
        area_center = {'area1': (x1, y1), 'area2': (x2, y2), ...}  # Centroid coordinates for each area
    """
    
    fig = plt.figure(figsize=(6,6))
    plt.axis('off')

    # Display the image
    plt.imshow(image, cmap="gray")
    
    # Iterate over each area and its corresponding data
    for area_name in areas_mask.keys():
        mask = areas_mask[area_name]
        histogram = areas_histogram[area_name]
        center = areas_center[area_name]
        contours, _ = cv2.findContours(mask, cv2.CV_32SC1, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Display the contours using red lines on the image.
            contour = contour.squeeze()
            plt.plot(contour[:, 0], contour[:, 1], 'red', linewidth=0.5)
            
            # Display the text.
            value = histogram.argmax() / 25
            plt.text(x=center[0], y=center[1], s=value,
                     size=6, color='purple', alpha=1, backgroundcolor = "white")
    
    # Save the figure to the specified output path.
    return fig



def save_json(areas_histogram, json_path):
    """
    Save the area_histogram dictionary as a JSON file at the specified output_path.
    """

    # np.ndarrays to lists
    for key, value in areas_histogram.items():
        areas_histogram[key] = value.tolist()

    # Write the JSON data to the file.
    with open(json_path, "w") as json_file:
        json.dump(json_path, json_file)


