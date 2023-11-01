import streamlit as st
from PIL import Image
import numpy as np
from pipeline import data_prepare, class_predict
import matplotlib.pyplot as plt
import numpy as np



st.header("Image Light Measure: ")
option = st.selectbox(label="Choose method: ",
                      options=["None", "Camera", "Upload File"])

col1, col2 = st.columns(2)

if option == "Upload File":
    file = col1.file_uploader(label="Please upload your face image.")
elif option == "Camera":
    file = col1.camera_input(label="Please take a photo")
elif option == "None":
    col1.text("Please, choose method.")   


if "file" in locals():
    try:
        pil_object = Image.open(file).convert("RGB")
        col1.image(pil_object)
        rgb_image = np.asarray(pil_object)
        x, fig_zones = data_prepare(rgb_image)
        y = round(class_predict(x), 2)

        class_1 = 100*y
        class_0 = 100-(100*y)
        col2.text(f"Probability of class 1 (good image): {class_1})")
        col2.text(f"Probability of class 0 (bad image): {class_0}")

        fig_pie, ax = plt.subplots()
        ax.pie([class_1, class_0], labels=["Class 1: good image",
                                           "Class 0: bad image"])
        col2.pyplot(fig_pie)
        col2.pyplot(fig_zones)

    except:
        "-"
    