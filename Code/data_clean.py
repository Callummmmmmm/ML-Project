#!/usr/bin/env python
# coding: utf-8

# In[1]:


from data_imports import pd


# In[12]:


def data():
    filepath = '/Users/callumwilson/Documents/GitHub/ML-Project/Input/20240423_CV_merge.csv'

    # Reading csv and assigning to a variable.
    # Skipping first 8 rows
    df_clean = pd.read_csv(filepath, skiprows = 8, index_col = 0)

    return df_clean

def time_based(input_data):
    df = input_data
    
    # Setting index to date-time format.
    df.index = pd.to_datetime(df.index)
    
    # Creating new variables based on date time.
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['time_since_start'] = (df.index - df.index.min()).days
    
    return df

