#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing essential libraries.
from data_imports import *

# Assigning filepath to variable.
# Sorting files in variable.
files = glob.glob("/mnt/scratch/projects/chem-acm-2018/ExtData/GEOS_0.5x0.625/MERRA2/*/*/*.A1.*")
files.sort()

# Assigning empty DataFrame to variable.
df = pd.DataFrame()

# Looping throuhg each file in files.
for file in files:
    print(file)
    
    # Opening each file.
    ds = xr.open_dataset(file)
    
    # Extracting the time variable.
    data = {'time': ds['time'].values}
    
    # Looping through each variable in dataset.
    for variable in ds.data_vars:
        
        # Select data for current variable at specific longitude and latitude.
        try:
            data[variable] = ds[variable].sel(lat=16.53, lon=-23.0, method='nearest').values
            
        # Print message if theres an error retrieving the data.
        except Exception as e:
            print(f"Could not retrieve data for variable {variable}: {e}")
    
    # Create a temporary DataFrame for the retrieved data.
    # Concatenate the temporary DataFrame with the main DataFrame
    temp_df = pd.DataFrame(data)
    df = pd.concat([df, temp_df], ignore_index=True)

# Save the concatenated DataFrame to a csv file in another folder.
df.to_csv(f'/users/dhd512/ML-Project/Input/all_variables.csv')

