# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:31:44 2022

@author: HP
"""

import streamlit as st
import numpy as np
import pickle
import os

#%% constant
MODEL_PATH = os.path.join(os.getcwd(),'best_estimator.pkl')
#%% loading
with open(MODEL_PATH,'rb') as file:
    model = pickle.load(file)
    

#%% test
#'age','cp','thall','thalachh','oldpeak'

new_data = np.expand_dims([65,3,1,152,3.2], axis= 0)
outcome = model.predict(new_data)[0]
print(outcome)


#%% app
 
st.subheader("Hi, How are you ? :wave:")
st.subheader("This App purpose is to predict the chance of a person having heart attack")
st.title('Prediction On Having A Heart Attack')

with st.form("heart attack prediction app"):
    age = st.number_input('Age of the patient',min_value= 0)
    cp = st.selectbox(" cp/Chest Pain type :  Value 0: typical angina, Value 1: atypical angina,  Value 2: non-anginal pain,  Value 3: asymptomatic" , (0,1,2,3))
    thall = st.selectbox("thall/thalassemia" , (1,2,3))
    thalachh = st.number_input('thalachh/maximum heart rate achieved',min_value= 0)
    oldpeak  = st.number_input('oldpeak/ST depression induced by exercise relative to rest',min_value= 0.0)
    
    
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        new_data = np.expand_dims([age,cp,thall,thalachh,oldpeak], axis= 0)
        outcome = model.predict(new_data)[0]
            
        if outcome == 0:
            st.write('< 50% diameter narrowing. less chance of heart disease :clap: :smile:')
            st.write('congrats you are healthy, keep it up')
            st.balloons()
        else:
            st.write('> 50% diameter narrowing. more chance of heart disease :grimacing:')
            st.write('start exercising now !!!!')
            

    

   
            