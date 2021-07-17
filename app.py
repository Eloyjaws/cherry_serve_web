import os
import torch
from PIL import Image
import streamlit as st
from collections import Counter
from torchvision import models, transforms
from htbuilder import HtmlElement, br

import numpy as np
from matplotlib import colors
from scipy.spatial import cKDTree as KDTree
from scipy.misc import face


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
        # [255, 255, 0],  # yellow
        # [0, 255, 255],  # cyan
        # [255, 0, 255],  # magenta
        [0, 0, 0],  # black
        [255, 255, 255],
    ]  # white

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
    print(Counter(pixels))

    return result


def getRipenessScoreNaive(image):
    pixels = image.getdata()
    print(Counter(pixels))


def getRipenessScoreKD(image):
    use_colors = {
        k: colors.cnames[k] for k in ["red", "green", "blue", "black", "purple"]
    }

    named_colors = {
        k: tuple(map(int, (v[1:3], v[3:5], v[5:7]), 3 * (16,)))
        for k, v in use_colors.items()
    }
    ncol = len(named_colors)

    ncol -= 1
    no_match = named_colors.pop("purple")

    color_tuples = list(named_colors.values())
    color_tuples.append(no_match)
    color_tuples = np.array(color_tuples)

    color_names = list(named_colors)
    color_names.append("no match")

    # get example picture
    img = face()

    # build tree
    tree = KDTree(color_tuples[:-1])
    # tolerance for color match `inf` means use best match no matter how
    # bad it may be
    tolerance = np.inf
    # find closest color in tree for each pixel in picture
    dist, idx = tree.query(img, distance_upper_bound=tolerance)
    # count and reattach names
    counts = dict(zip(color_names, np.bincount(idx.ravel(), None, ncol + 1)))

    print(counts)


st.markdown(br(), unsafe_allow_html=True)
file_up = st.file_uploader("Upload an image", type="jpg")
st.markdown(br(), unsafe_allow_html=True)


if file_up:
    image = Image.open(file_up)

    image = transform(image)
    image = torch.unsqueeze(image, 0)
    model.eval()
    output = model(image)

    in_image = tensor_to_PIL(image * 0.5 + 0.5)
    out_image = tensor_to_PIL(output * 0.5 + 0.5)

    im3 = getRipenessScore(out_image)

    col1, col2, col3 = st.beta_columns(3)
    # col1, col2 = st.beta_columns(2)

    col1.header("Original")
    col1.image(in_image, use_column_width=True)

    col2.header("Mask")
    col2.image(out_image, use_column_width=True)

    col3.header("Rounded pixels")
    col3.image(im3, use_column_width=True)
