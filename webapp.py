#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from PIL import Image


# In[3]:


st.write('''
# visualizing stock data
**Karami milad**
''')
img=Image.open('C:/Users/lenovo/trade.jpg')
st.image(img,width=600,caption='Karami Milad')
st.sidebar.header('Input Data')


# In[4]:


def get_input():
    numb=st.sidebar.text_input('N Last Days',50)
    stock_symbol=st.sidebar.text_input('insert stock symbol','FOOLAD')
    return stock_symbol , numb


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
        'NONE'


# In[10]:


def get_data(symbol,n):
    if symbol.upper()=='FOOLAD':
        df=pd.read_csv('C:/Users/lenovo/foolad.csv')
    elif symbol.upper()=='KHODRO':
        df=pd.read_csv('C:/Users/lenovo/khodro.csv')
    elif symbol.upper()=='AMZN':
        df=pd.read_csv('C:/Users/lenovo/amzn.csv')
    elif symbol.upper()=='TSLA':
        df=pd.read_csv('C:/Users/lenovo/tsla.csv')
    else:
        df=pd.DataFrame(columns=['Date','First','High','Low','Close','Value','Volume','Openint','Per','Open','last'])
        
    df=df.set_index(pd.DatetimeIndex(df['Date'].values))
    n= int(n)
    df=df.head(n)
    return df


# In[11]:


symbol , n = get_input()
df=get_data(symbol,n)
company=get_company_name(symbol.upper())
st.header(company + ' Close Price\n')
st.line_chart(df['Close'])
st.header(company + ' Volume\n')
st.line_chart(df['Volume'])
st.header('Stock Datas')
st.write(df.describe())


# In[ ]:




