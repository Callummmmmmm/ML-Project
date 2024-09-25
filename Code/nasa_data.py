#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import xarray as xr
import glob
import pandas as pd

files=glob.glob("/mnt/scratch/projects/chem-acm-2018/ExtData/GEOS_0.5x0.625/MERRA2/*/*/*.A1.*")

first=True
files.sort()

variable='TS'
for file in files:
        print(file)
        ds=xr.open_dataset(file)

        if(first):
                first=False
                df=pd.DataFrame({'time':ds['time'].values,
                                                 'TS':ds[variable].sel(lat=16.53, lon=-23.0, method='nearest').values})
        else:
                df=pd.concat([df, pd.DataFrame({'time':ds['time'].values,
                                                 'TS':ds[variable].sel(lat=16.53, lon=-23.0, method='nearest').values})],
                                                 ignore_index=True)

df.to_csv(f'/mnt/ML-Project/Input/{variable}.csv')

