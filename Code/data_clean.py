#!/usr/bin/env python
# coding: utf-8

# In[1]:


from data_imports import pd


# In[2]:


def CVAO_data():
    filepath = '/Users/callumwilson/Documents/GitHub/ML-Project/Input/20240423_CV_merge.csv'

    # Reading csv and assigning to a variable.
    # Skipping first 8 rows
    df_clean = pd.read_csv(filepath, skiprows = 8, index_col = 0)
    
    # Setting index to date-time format.
    df_clean.index = pd.to_datetime(df_clean.index)

    return df_clean

def GEOS_FP_A1_data():
    filepath = '/Users/callumwilson/Documents/GitHub/ML-Project/Input/all_variables.csv'
    
    df_clean = pd.read_csv(filepath)
    
    df_clean.drop(columns='Unnamed: 0', inplace=True)
    df_clean.rename(columns={'time': 'Date and Time (UTC)'}, inplace=True)
    df_clean.set_index('Date and Time (UTC)', inplace=True)
    df_clean.index = pd.to_datetime(df_clean.index)
    df_clean.index = df_clean.index + pd.Timedelta(minutes=30)

    return df_clean

