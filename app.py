import io
import streamlit as st
import requests
import numpy as np
from PIL import Image
from model import get_caption_model, generate_caption


@st.cache(allow_output_mutation=True)
def get_model():
    return get_caption_model()

caption_model = get_model()


def predict():
    pred_caption = generate_caption('tmp.jpg', caption_model)

    st.write('Predicted Captions:')
    st.write(pred_caption)

    for _ in range(4):
        pred_caption = generate_caption('tmp.jpg', caption_model, add_noise=True)
        st.write(pred_caption)


img_url = st.text_input(label='Enter Image URL')

if (img_url != "") and (img_url != None):
    img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.convert('RGB')
    st.image(img)
    img.save('tmp.jpg')
    st.image(img)
    predict()


img = st.file_uploader(label='Upload Image', type=['jpg', 'png'])

if img != None:
    img = img.read()
    img = Image.open(io.BytesIO(img))
    img = img.convert('RGB')
    img.save('tmp.jpg')
    st.image(img)
    predict()
