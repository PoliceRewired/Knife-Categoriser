import streamlit as st
import numpy as np
import pandas as pd
from fastai.vision.all import *
from fastai.vision.widgets import *
import os
from ftplib import FTP

#define our environment variables (secret stuff I don't want on github)
hostname = os.getenv("hostname")
username = os.getenv("username")
password = os.getenv("password")

#import our trained learning mode
learn_inf = load_learner('export.pkl')

result = "Awaiting Upload"
result_detail = ""

#couple of dictionaries for later
results_dict = {"butterfly":["butterfly knife","A balisong, also known as a fan knife, butterfly knife or Batangas knife, is a type of folding pocketknife that originated in the Philippines. Its distinct features are two handles counter-rotating around the tang such that, when closed, the blade is concealed within grooves in the handles."],
                "pocket":["pocket knife", "A pocketknife is a foldable knife with one or more blades that fit inside the handle that can still fit in a pocket."],
               "machete":["machete", "A machete is a broad blade used either as an agricultural implement similar to an axe, or in combat like a long-bladed knife."],
               "bayonet":["bayonet/combat knife", "A combat knife is a fighting knife designed solely for military use and primarily intended for hand-to-hand or close combat fighting."],
               "kitchen":["kitchen knife", "A household knife intended for cooking."]
}

correct_options_dict = {"Butterfly Knife":"butterfly", "Pocket Knife":"pocket", "Machete":"machete", "Bayonet or Combat Knife":"bayonet", "Kitchen Knife":"kitchen"}
st.set_page_config(page_title="Knife Classifier")
st.title('Knife Classifier')
st.markdown('''Welcome to our prototype Knife Classifier. This tool is currently work-in-progress while we train the model, and will be tweaked on a weekly basis or depending on demand. 
The model is trained on the below categories:
- Butterfly Knives
- Folding Pocket Knives
- Machetes
- Combat and Bayonet Knives
- Kitchen Knives ''')

st.write("This model expects pictures single knife pictures, so remove or crop out background items (such as notebooks and rulers)")

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
    correct_button = st.radio("Did we get this right?", ["Yes", "No"])
    if correct_button == "Yes":
        st.write("We're always trying to improve...can we use this picture to improve our results?")
        if st.button('Send Us The Picture!'):
            st.write("Thanks for the feedback! We'll try harder to get it right next time.")
            with FTP(hostname, username, password) as ftp:
                ftp.cwd(result)
                ftp.storbinary(f'STOR {uploaded_file.name}', uploaded_file)
    if correct_button == "No":
        correct_option = st.radio("Oh no! I'm always trying to improve...can you tell me which of the below it was?",["Butterfly Knife", "Pocket Knife", "Machete", "Bayonet or Combat Knife","Kitchen Knife", "Other"])
        if uploaded_file is not None:
            if correct_option == "Other":
                st.write("Sadly, it doesn't look like we recognise these yet, but we're always improving!")
            else:
                st.write("We're always trying to improve...can we use this picture to improve our results?")
                if st.button('Send Us The Picture!'):
                    st.write("Thanks for the feedback! We'll try harder to get it right next time.")
                    with FTP(hostname, username, password) as ftp:
                        ftp.cwd(correct_options_dict[correct_option])
                        ftp.storbinary(f'STOR {uploaded_file.name}', uploaded_file)


