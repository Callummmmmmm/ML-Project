#!/usr/bin/env python
# coding: utf-8

# In[1]:


from data_imports import pd


# In[6]:


def data():
    filepath = '/Users/callumwilson/Documents/GitHub/ML-Project/Input/20240423_CV_merge.csv'

    # Reading csv and assigning to a variable.
    # Skipping first 8 rows
    df_clean = pd.read_csv(filepath, skiprows = 8, index_col = 0)

    return df_clean

