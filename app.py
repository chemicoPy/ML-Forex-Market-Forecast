#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from torch.utils.data import Dataset
#from torch.utils.data import DataLoader

import streamlit as st
import time
import requests
import pandas as pd
import numpy as np
import re
import string
import os
import json
import streamlit.components.v1 as components
from io import BytesIO
from time import sleep
import math
from pathlib import Path
import numpy as np
from numpy import *
import pandas as pd
import json
from pandas import DataFrame, Series
from numpy.random import randn
import requests
import io
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


#st.write("All libraries loaded")


# Desiging & implementing changes to the standard streamlit UI/UX
st.set_page_config(page_icon="img/page_icon.png")    #Logo
st.markdown('''<style>.css-1egvi7u {margin-top: -4rem;}</style>''',
    unsafe_allow_html=True)
# Design change hyperlink href link color
st.markdown('''<style>.css-znku1x a {color: #9d03fc;}</style>''',
    unsafe_allow_html=True)  # darkmode
st.markdown('''<style>.css-znku1x a {color: #9d03fc;}</style>''',
    unsafe_allow_html=True)  # lightmode
# Design change height of text input fields headers
st.markdown('''<style>.css-qrbaxs {min-height: 0.0rem;}</style>''',
    unsafe_allow_html=True)
# Design change spinner color to primary color
st.markdown('''<style>.stSpinner > div > div {border-top-color: #9d03fc;}</style>''',
    unsafe_allow_html=True)
# Design change min height of text input box
st.markdown('''<style>.css-15tx938{min-height: 0.0rem;}</style>''',
    unsafe_allow_html=True)

# Design hide top header line
hide_decoration_bar_style = '''<style>header {visibility: hidden;}</style>'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
# Design hide "made with streamlit" footer menu area
hide_streamlit_footer = """<style>#MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_footer, unsafe_allow_html=True)

# disable warnings
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)



#Data & model configuration

config = {
    
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
    },
    "plots": {
        "show_plots": True,
        "xticks_interval": 90,
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1,
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu", # "cuda" or "cpu"
        "batch_size": 64,
        "num_epoch": 200,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}



#Intervals of interest: Monthly, Weekly, Daily, 4hour, 1hour, 30 minutes

# API Valid periods: 1m,5m,15m,30m,1h,2h,4h,5h,1d,1w,month


#df = df.drop(columns=["index", "t", "v"], axis = 1, inplace = True)
#df.rename(columns = {"c":"close", "l":"low", "o":"open","h":"high"}, inplace = True)
#df.time = pd.to_datetime(df.time, unit='ms')
#df = df.sort_values(by='time',ascending=True)
#print(df.head())
#print("Size of data:", len(df))


import os 

st.title('ML-Forex-Market-Forecast web app')
st.subheader("Navigate to side bar to see full project info")
    
    # ------ layout setting---------------------------

st.sidebar.markdown(
            """
     ----------
    ## Project Overview
    ML-Forex-Market-Forecast is a ...
    """)    

   
st.sidebar.markdown("## Select Forex pair & Interval below") # add a title to the sidebar container
    
    # ---------------forex pair selection------------------
  
symb = st.sidebar.selectbox(
        '', ["Select Forex Pair of interest", "XAU/USD","BTC/USD","ETH/USD","DOGE/USD"], index=0)
  
time_int = st.sidebar.selectbox(
        '', ["Interval of interest", "1m","5m","15m","30m","1h","2h","4h","5h","1d","1w", "month"], index=0)
 
st.sidebar.markdown(
    """
    -----------
    # Let's connect
    
    [![Victor Ogunjobi](https://img.shields.io/badge/Author-@VictorOgunjobi-gray.svg?colorA=gray&colorB=dodgergreen&logo=github)](https://www.github.com/chemicopy)
    
    [![Victor Ogunjobi](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logoColor=white)](https://www.linkedin.com/in/victor-ogunjobi-a761561a5/)
    
    [![Victor Ogunjobi](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=gray)](https://twitter.com/chemicopy_)
    """)
   
            
st.write("\n")  # add spacing    
   

#API_URL = "https://fcsapi.com/api-v3/forex/history?symbol="+symb+"&period="+time_int+"&access_key=OePoBiGZhsN57a4OYrFH&level=3"

API_URL = "https://fcsapi.com/api-v3/forex/history?symbol="+symb+"&period="+time_int+"&access_key=OePoBiGZhsN57a4OYrFH&level=3"
r = requests.get(API_URL)
resp = "response"
json = r.json()
df = pd.DataFrame(json[resp]).T
df.reset_index(inplace=True)
df['c'] = pd.to_numeric(df['c'], errors='coerce')
    
def download_data(config, plot=False):
    #Date and reverse - confirm if it was reversed or not in the sample script i'm using. if it was, use the commented version in the next line
    data_date = list(df.tm)
    #data_date = list(df.time[::-1])
    # Close price and reverse - confirm if it was reversed or not in the sample script i'm using. if it was, use the commented version in the next line
    data_close_price = df.c
    #data_close_price = df.close[::-1]
    data_close_price = np.array(data_close_price)
    
    # Data points
    num_data_points = len(data_date)
    #display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
    #print("Number data points:", num_data_points, display_date_range)
    
    if plot:
        fig = figure(figsize=(25, 5), dpi=80)
        fig.patch.set_facecolor((1.0, 1.0, 1.0))
        plt.plot(data_date, data_close_price, color=config["plots"]["color_actual"])
        xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
        x = np.arange(0,len(xticks))
        plt.xticks(x, xticks, rotation='vertical')
        plt.title("Close price plot for " + "the data" + ", ")
        plt.grid(b=None, which='major', axis='y', linestyle='--')
        plt.show()
        st.write(fig)

    return data_date, data_close_price, num_data_points

data_date, data_close_price, num_data_points = download_data(config, plot=config["plots"]["show_plots"])



   
