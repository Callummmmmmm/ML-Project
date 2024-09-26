#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import xarray as xr
import glob
import pandas as pd

files = glob.glob("/mnt/scratch/projects/chem-acm-2018/ExtData/GEOS_0.5x0.625/MERRA2/*/*/*.A1.*")
files.sort()

df = pd.DataFrame()

for file in files:
    print(file)
    ds = xr.open_dataset(file)
    
    data = {'time': ds['time'].values}
    
    for variable in ds.data_vars:
        try:
            data[variable] = ds[variable].sel(lat=16.53, lon=-23.0, method='nearest').values
        except Exception as e:
            print(f"Could not retrieve data for variable {variable}: {e}")
    
    temp_df = pd.DataFrame(data)
    df = pd.concat([df, temp_df], ignore_index=True)

df.to_csv(f'/users/dhd512/ML-Project/Input/all_variables.csv')

