#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image


# In[2]:


st.write('''
# STOCK VISUALIZER

**Karami Milad**

''')
img= Image.open('C:/Users/lenovo/trade.jpg')
st.image(img,width=600)


# In[3]:


st.sidebar.header('INSERT DATA')
def data():
    n=st.sidebar.text_input('How many days you wanna predict? ',5)
    symbol=st.sidebar.selectbox('Select The Symbol : ',['FOOLAD','AMZN','KHODRO','TSLA'])
    return n , symbol


# In[4]:


def get_data():
    if symbol=='FOOLAD':
        df=pd.read_csv('C:/Users/lenovo/foolad.csv')
    elif symbol=='KHODRO':
        df=pd.read_csv('C:/Users/lenovo/khodo.csv')
    elif symbol=='AMZN':
        df=pd.read_csv('C:/Users/lenovo/amzn.csv')
    elif symbol=='TSLA':
        df=pd.read_csv('C:/Users/lenovo/tsla.csv')
        
    df=df.set_index(pd.DatetimeIndex(df['Date'].values)) 
    return df


# In[5]:


def get_company_name(symbol):
    if symbol=='FOOLAD':
        return 'FOOLAD'
    elif symbol== 'KHODRO':
        return 'KHODRO'
    elif symbol== 'AMZN':
        return 'AMAZON'
    elif symbol=='TSLA':
        return 'TESLA'
    else :
        return  'NONE'


# In[6]:


n , symbol = data()
df=get_data()
company=get_company_name(symbol)
st.header(company + 'Close Price\n')
st.line_chart(df['Close'])
st.header(company + 'Volume\n')
st.line_chart(df['Volume'])
st.header('Stock Datas')
st.write(df.describe())


# In[7]:


df=df[['Close']]
forecast=int(n)
df['Prediction']=df[['Close']].shift(-forecast)
x= np.array(df.drop(['Prediction'],1))
x= x[:-forecast]
y= np.array(df['Prediction'])
y=y[:-forecast]


# In[8]:


xtrain , xtest , ytrain , ytest=train_test_split(x,y,test_size=0.2)
mysvr=SVR(kernel='rbf',C=1000,gamma=0.1)
mysvr.fit(xtrain,ytrain)
svmconf=mysvr.score(xtest,ytest)
st.header('SVM Accuracy')
st.success(svmconf)


# In[9]:


x_forecast=np.array(df.drop(['Prediction'],1))[-forecast:]
svmpred=mysvr.predict(x_forecast)
st.header('SVM Prediction')
st.success(svmpred)


# In[10]:


lr=LinearRegression()
lr.fit(xtrain,ytrain)
lrconf=lr.score(xtest,ytest)
st.header('LR Accuracy')
st.success(lrconf)


# In[11]:


lrpred=lr.predict(x_forecast)
st.header('LR Prediction')
st.success(lrpred)


# In[ ]:




