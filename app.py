import os
import torch
from PIL import Image
import streamlit as st
from torchvision import models, transforms
from htbuilder import HtmlElement, br

hide_streamlit_style = """
<style>
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


st.markdown(br(), unsafe_allow_html=True)
file_up = st.file_uploader("Upload an image", type="jpg")
st.markdown(br(), unsafe_allow_html=True)


if file_up:
    image = Image.open(file_up)
    batch_t = torch.unsqueeze(transform(image), 0)
    model.eval()
    out = model(batch_t)

    out_norm = out[0] * 0.5 + 0.5
    out_image = tensor_to_PIL(out_norm)

    col1, col2 = st.beta_columns(2)

    col1.header("Original")
    col1.image(image, use_column_width=True)

    col2.header("Mask")
    col2.image(out_image, use_column_width=True)
