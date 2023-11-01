import streamlit as st
import pickle
import pandas as pd
import numpy as np
from mputils import FaceDetector
from PIL import Image
# from sutils import 



def data_pipeline(image):
    """Numpy Image -> X for prediciton"""
    face_detector = FaceDetector('models/face_landmarker.task')
    _, _, areas_histogram = face_detector.image_pipeline(image)
    array_1d = np.array(list(areas_histogram.values())).flatten() # values to flat array
    array_1d = array_1d.reshape(1, -1)
    return array_1d

def class_predict(array_1d, return_type="proba"):
    logreg = pickle.load(open("models/logreg.sav", "rb"))
    if return_type == "class":
        return logreg.predict(array_1d)
    elif return_type == "proba":
        # returns class 1 probability
        return logreg.predict_proba(array_1d)[0][1]


pil_object = Image.open("data/0000_00000001_illu_cafe-inside-3_extracted.jpg").convert("RGB")
rgb_image = np.asarray(pil_object)
x = data_pipeline(rgb_image)
print(class_predict(x))