import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from fastai.vision.all import *
from fastai.vision.widgets import *



learn_inf = load_learner('export.pkl')

result = "Awaiting Upload"
result_detail = ""

results_dict = {"butterfly":["Butterfly Knife","A balisong, also known as a fan knife, butterfly knife or Batangas knife, is a type of folding pocketknife that originated in the Philippines. Its distinct features are two handles counter-rotating around the tang such that, when closed, the blade is concealed within grooves in the handles."],
                "pocket":["Pocket Knife", "A pocketknife is a foldable knife with one or more blades that fit inside the handle that can still fit in a pocket."],
               "machete":["Machete", "A machete is a broad blade used either as an agricultural implement similar to an axe, or in combat like a long-bladed knife. The blade is typically 30 to 45 centimetres (12 to 18 in) long and usually under 3 millimetres (0.12 in) thick."],
               "bayonet":["Bayonet or Combat Knife", "A combat knife is a fighting knife designed solely for military use and primarily intended for hand-to-hand or close combat fighting.[1][2][3]"],
               "kitchen":["Kitchen knife", "A household knife intended for cooking"]
}
st.title('Knife Classifier')
st.write("Upload your knife picture below and click confirm to begin classification")

uploaded_file = st.file_uploader("Your Knife Picture", type=["png","jpg","bmp", "tiff","gif","eps","raw","jpeg",], accept_multiple_files=False, key="knife_pic")

if uploaded_file is not None:
    knife_img = PILImage.create((uploaded_file))
    pred, pred_idx, probs = learn_inf.predict(knife_img)
    result = "You uploaded a " + results_dict[pred][0]
    result_detail = results_dict[pred][1]

if uploaded_file is not None:
    st.image(knife_img, width=150)

st.write(result)
st.write(result_detail)