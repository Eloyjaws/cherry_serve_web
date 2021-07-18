import os
import torch
import numpy as np
from PIL import Image
import streamlit as st
from collections import Counter
from torchvision import models, transforms
from htbuilder import HtmlElement, br

st.set_page_config(
    page_title="Technoserve - Ripeness Predictor",
    page_icon="favicon.ico",
    initial_sidebar_state="auto",
)

hide_streamlit_style = """
    <style>
    header > div:first-child {
    background-image: linear-gradient(90deg, rgb(246, 51, 102), rgb(46 154 255), rgb(9 171 59));
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.write(
    """
    # ML for Coffee Cherry Ripeness Prediction
    Upload an image and get the coffee cherry ripeness score
    """
)

model = torch.jit.load("benchmark.ptl", map_location="cpu")

transform = transforms.Compose(
    [
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

unloader = transforms.ToPILImage()


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def distance(col1, col2):
    r1, g1, b1 = col1
    r2, g2, b2 = col2
    return (r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2


# Adapted from  https://stackoverflow.com/questions/50545192/count-different-colour-pixels-python
def getRipenessScore(im):
    # All colours in the image will be forced to nearest one in this list
    refColours = [
        [255, 0, 0],  # red
        [0, 255, 0],  # green
        [0, 0, 255],  # blue
        [0, 0, 0],  # black
        [255, 255, 255],  # white
    ] 

    imrgb = im.convert("RGB")
    (w, h) = im.size[0], im.size[1]

    # Make a buffer for our output pixels
    px = np.zeros(w * h, dtype=np.uint8)
    idx = 0

    for pixel in list(imrgb.getdata()):
        # Calculate distance from this pixel to first one in reference list
        mindist = distance([pixel[0], pixel[1], pixel[2]], refColours[0])
        nearest = 0
        # Now see if any other colour in reference list is closer
        for index in range(1, len(refColours)):
            d = distance([pixel[0], pixel[1], pixel[2]], refColours[index])
            if d < mindist:
                mindist = d
                nearest = index
        # Set output pixel to nearest
        px[idx] = nearest
        idx += 1

    # Reshape our output pixels to match input image
    px = px.reshape(w, h)
    # Make output image from our pixels
    outimg = Image.fromarray(px).convert("P")
    # Put our palette of favourite colours into the output image
    palette = [item for sublist in refColours for item in sublist]
    outimg.putpalette(palette)
    result = outimg.convert("RGB")

    pixels = result.getdata()
    return Counter(pixels)

st.markdown(br(), unsafe_allow_html=True)
file_up = st.file_uploader("Upload an image", type="jpg")

if file_up:
    image = Image.open(file_up)

    image = transform(image)
    image = torch.unsqueeze(image, 0)
    model.eval()
    output = model(image)

    in_image = tensor_to_PIL(image * 0.5 + 0.5)
    out_image = tensor_to_PIL(output * 0.5 + 0.5)

    frequencies = getRipenessScore(out_image)

    col1, col2 = st.beta_columns(2)

    col1.header("Original")
    col1.image(in_image, use_column_width=True)

    col2.header("Mask")
    col2.image(out_image, use_column_width=True)

    refColours = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
    }

    redCount = frequencies.get(refColours.get("red"))
    blueCount = frequencies.get(refColours.get("blue"))
    greenCount = frequencies.get(refColours.get("green"))

    totalCount = redCount + blueCount + greenCount
    ripenessScore = (redCount / totalCount) * 100
    overripenessScore = (blueCount / totalCount) * 100
    underripenessScore = (greenCount / totalCount) * 100

    st.markdown(br(), unsafe_allow_html=True)
    res1, res2, res3 = st.beta_columns(3)

    res1.success(f"UnderRipe: {underripenessScore:.2f}%")
    res2.info(f"OverRipe: {overripenessScore:.2f}%")
    res3.error(f"Ripe: {ripenessScore:.2f}%")
