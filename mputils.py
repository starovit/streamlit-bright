import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import cv2


class FaceDetector():

    AREA_IDS = {"lips_left": [0, 17, 57],
            "eye_left": [226, 223, 189, 121],
            "nose_left": [8, 94, 102, 189, 8],
            "top1_left": [10, 8, 189, 223, 103, 67, 109, 10],
            "top2_left": [223, 225, 226, 127, 162, 21, 54, 103, 223],
            "bot1_left": [152, 17, 43, 57, 150, 149, 176, 148, 152],
            "bot2_left": [57, 150, 172, 132, 57],
            "mid1_left": [102, 132, 57, 0, 94],
            "mid2_left": [132, 127, 226, 121, 102],
            
            "lips_right": [0, 17, 287],
            "eye_right": [443, 446, 350, 413],
            "nose_right": [8, 94, 331, 413, 8],
            "top1_right": [10, 8, 413, 443, 332, 297, 338, 10],
            "top2_right": [443, 445, 446, 356, 389, 251, 284, 332, 443],
            "bot1_right": [152, 17, 273, 287, 379, 378, 400, 377, 152],
            "bot2_right": [287, 397, 361, 379, 287],
            "mid1_right": [331, 361, 287, 0, 94],
            "mid2_right": [350, 446, 356, 361, 331],
           }

    def __init__(self, model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)


    def detect(self, rgb_image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detection_result = self.detector.detect(mp_image)
        if detection_result.face_landmarks:
            return detection_result.face_landmarks[0]
        else:
            return None
    

    def landmarks_to_coordinates(self, landmarks, image_shape):
        landmarks_coordinates = []
        for landmark in landmarks:
            landmarks_coordinates.append([landmark.x*image_shape[1],
                                          landmark.y*image_shape[0]])
        return np.array(landmarks_coordinates).astype(int)
    
    def find_histograms(self,
                     bright_image,
                     absolute_coordinates,
                     area_ids):
        
        areas_mask = {}
        areas_center = {}
        areas_histogram = {}

        for area_name, ids in area_ids.items():
            area_coords = absolute_coordinates[ids, ::]

            hull = cv2.convexHull(area_coords)
            mask = np.zeros(bright_image.shape)
            mask = cv2.drawContours(mask, [hull], -1, 1, thickness=cv2.FILLED)
            masked = bright_image * mask

            areas_mask[area_name] = mask.astype(int)
            areas_center[area_name] = area_coords.mean(axis=0).astype(int)
            areas_histogram[area_name] = calc_hist(masked)

        return areas_mask, areas_center, areas_histogram
    

    def image_pipeline(self, rgb_image):
        bright_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)[:,:,-1]
        landmarks = self.detect(rgb_image)
        abs_coords = self.landmarks_to_coordinates(landmarks, bright_image.shape)
        areas_mask, areas_center, areas_histogram = \
                self.find_histograms(bright_image=bright_image,
                                    absolute_coordinates=abs_coords,
                                    area_ids=FaceDetector.AREA_IDS)

        return areas_mask, areas_center, areas_histogram
    
    def read_image(self, image_path):
        rgb_image = cv2.imread(image_path)[:,:,::-1].copy()
        return rgb_image





def calc_hist(array, bin_num=10):
    """
    Convert a 2D NumPy array to a normalized histogram.

    Parameters:
        array (numpy.ndarray): A 2D NumPy array.
        bin_num (int): Number of bins for the histogram. Default is 10.

    Returns:
        numpy.ndarray: A normalized histogram.
    """

    flat = array.flatten()
    flat = flat[flat!=0]
    hist, bin_edges = np.histogram(flat, bins=range(0,260,bin_num))
    hist_norm = hist/hist.sum()
    hist_norm = hist_norm.round(3)
    return hist_norm