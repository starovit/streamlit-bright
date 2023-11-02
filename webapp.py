import streamlit as st
from PIL import Image
import numpy as np
from pipeline import data_prepare, class_predict
import matplotlib.pyplot as plt
import numpy as np

st.header("Image Light Measure: ")

def main_loop(file):
    pil_object = Image.open(file).convert("RGB")
    col1.image(pil_object)
    rgb_image = np.asarray(pil_object)
    x, fig_zones = data_prepare(rgb_image)
    y = round(class_predict(x), 2)

    class_1 = 100*y
    class_0 = 100-(100*y)
    col2.text(f"Probability of class 0 (bad image): {class_0}")
    col2.text(f"Probability of class 1 (good image): {class_1})")

    fig_bar, ax = plt.subplots()
    ax.bar(x=[f"Bad light: {class_0}%", f"Good light: {class_1}%"],
            height=[class_0, class_1], color=["red", "green"])
    col2.pyplot(fig_bar)
    col2.pyplot(fig_zones)


option = st.selectbox(label="Choose method: ",
                      options=["None", "Camera", "File"])

col1, col2 = st.columns(2)

if option == "None":
    file = None
if option == "Camera":
    file = col1.camera_input(label="Please take a photo")
    col2.text(type(file))
elif option == "File":
    file = col1.file_uploader(label="Please upload face image")

if file is not None:
    main_loop(file)