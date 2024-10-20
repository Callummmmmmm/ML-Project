#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Importing essential libraries.
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import numpy as np
import shap
import xarray as xr
import glob
import seaborn as sns
