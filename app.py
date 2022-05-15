import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings


st.beta_set_page_config(page_title="The Crop Recommender System", page_icon="🌿", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> 🌾 My Crop Recommendation System 🌾 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1,col2  = st.beta_columns([2,2])
    
    with col1: 
        with st.beta_expander(" ℹ️ Information", expanded=True):
            st.write("""
           For our Risk Management System, we are trying to implement a crop recommandation system. In fact, the farmer needs to fill the informations gathered or the informations he know regarding the crop, then he could get an honest and good recommandation about what type of vegetables or fruits to use
            """)
        '''
        ## If you want to know how it's working ❓ 
        Just fill the parameters and our machine learning model (the best one of six models) will predict the most suitable crops to grow in the farm
        '''


    with col2:
        st.subheader(" 👨‍🌾 Find out the most suitable crop to grow in your farm 👨‍🌾")
        N = st.number_input("Nitrogen", 1,10000)
        P = st.number_input("Phosporus", 1,10000)
        K = st.number_input("Potassium", 1,10000)
        temp = st.number_input("Temperature",0.0,100000.0)
        humidity = st.number_input("Humidity in %", 0.0,100000.0)
        ph = st.number_input("Ph", 0.0,100000.0)
        rainfall = st.number_input("Rainfall in mm",0.0,100000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1,-1)
        
        if st.button('Predict'):

            loaded_model = load_model('model.pkl')
            prediction = loaded_model.predict(single_pred)
            col1.write('''
		    ## Results 🔍 
		    ''')
            col1.success(f"{prediction.item().title()} are recommended by our system for your farm.")

    st.warning("Note: This module included in our system is used in the context of my PFE. Made by Hajar Lachheb")
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
	main()
