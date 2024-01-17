#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install Streamlit
# !pip install Pillow


# In[1]:


import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[2]:


st.header('Diabetes Prediction App')


# In[3]:


im=Image.open('C:\\Users\\User\\Downloads\\diab.jpg')


# In[4]:


st.image(im)


# In[5]:


data=pd.read_csv('diabetes.csv')


# In[11]:


data.head()


# In[7]:


st.subheader('Data')


# In[8]:


st.dataframe(data)


# In[9]:


st.subheader('Data Info')


# In[10]:


st.write(data.iloc[:,:-1].describe())


# # Model

# In[12]:


x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[13]:


model=RandomForestClassifier(n_estimators=500)


# In[14]:


model.fit(x_train,y_train)


# In[15]:


y_pred=model.predict(x_test)


# In[19]:


acc=accuracy_score(y_test,y_pred)


# In[20]:


st.subheader('Accuracy of the model')


# In[21]:


st.write(acc)


# In[24]:


# st.text_input(label='Enter Your Age')
# st.text_area(label='What is the age?')


# In[28]:


def user_input():
    
    preg=st.slider('Pregnancy',0,20,0)
    gluc=st.slider('Glucose',0,200,0)
    bp=st.slider('Blood Pressure',0,130,0)
    sthick=st.slider('Skin Thickness',0,100,0)
    ins=st.slider('Insulin',0.0,1000.0,0.0)
    bmi=st.slider('BMI',0.0,70.0,0.0)
    dpf=st.slider('Diabetes Prediction Function',0.000,3.000,0.000)
    age=st.slider('Age',0,120,0)
    
    input_dict={'Pregnancy':preg,
               'Glucose':gluc,
               'Blood Pressure':bp,
               'Skin Thickness':sthick,
               'Insulin':ins,
               'BMI':bmi,
               'Diabetes_Prediction_Function':dpf,
               'Age':age}
    
    return pd.DataFrame(input_dict,index=['User'])


# In[29]:


st.subheader('Enter Your Data')
ui=user_input()
st.write(ui)


# In[30]:


pred=model.predict(ui.values)
st.subheader('Predictions (0-Non Diabetic 1-Diabetic)')
st.write(pred)


# In[ ]:


# preg=st.slider('Pregnancy',0,20,0)
# gluc=st.slider('Glucose',0,200,0)
# bp=st.slider('Blood Pressure',0,130,0)
# sthick=st.slider('Skin Thickness',0,100,0)
# ins=st.slider('Insulin',0.0,1000.0,0.0)
# bmi=st.slider('BMI',0.0,70.0,0.0)
# dpf=st.slider('Diabetes Prediction Function',0.000,3.000,0.000)
# age=st.slider('Age',0,120,0)

