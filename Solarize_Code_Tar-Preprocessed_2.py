#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Code to get Clear Sky Radiation per hour 
# To check against the MET Office Data 

from pysolar.solar import *
import datetime

#Coventry Co-ordinates
latitude = 52.424
longitude = -1.536

hours_total = 24 
days_total = 28
year = 2020
month = 1
day = 1



for hour in range(hours_total):
    date = datetime.datetime(year, month, day, hour, 13, 1, 130320, tzinfo=datetime.timezone.utc)

    Sun_altitude = get_altitude(latitude, longitude, date)
    # print("Sun_Altitude : %f" % (Sun_altitude))

    Azimuth = get_azimuth(latitude, longitude, date)
    # print("Azimuth : %f" % (Azimuth))

    altitude_deg = get_altitude(latitude, longitude, date)
    radiation.get_radiation_direct(date, altitude_deg)
    # print("Hours: %d, Radiation : %f" % (hour,radiation.get_radiation_direct(date, altitude_deg)))


# In[2]:


# OLD Code
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from datetime import datetime

cut_off = None

fileName = "C:/Users/tartn/Documents/Solarize_Data/Coventry/midas-open_uk-radiation-obs_dv-202107_west-midlands_24102_coventry-coundon_qcv-1_2020.csv"
df = pd.read_csv(fileName, skiprows=list(range(0, 75)))
# when reading this csv files, switch the '\' to '/' forward slashes
# since file data frame is uneven, we need to skip the first 0-75 rows to access the real data 
year = 'data'

year = pd.DataFrame(columns=['Time','Irradiance','Effective Irradiance']) 
year['Time'] = df['ob_end_time']
year['Irradiance'] = df['glbl_irad_amt']
year.drop(df.tail(1).index,inplace=True) # remove the last 'end' value 

year = year[:cut_off]

# iterate through the entire'Time' column 
# subtract the current row with the starting date 
# get the number of seconds elapsed and divide by 3600 to get hour 
for index, row in year.iterrows():
    year.loc[index,'hours'] = (datetime.strptime(row['Time'],'%Y-%m-%d %H:%M:%S') - datetime.strptime(year.iloc[0]['Time'],'%Y-%m-%d %H:%M:%S')).total_seconds() / 3600

# since irradiance is in KJ/m2 per hour, we need to divide it down to W/m2
year['Irradiance'] = (year['Irradiance']*1000)/3600

######data['Effective Irradiance'] = formula zenith azimuth times irradiance data  

year.head()
print(year.Time.size)

# 365*24  = 8760
# since there is 2020 is  leap year, there are 366 days = 8784 hours
# 2020 - 8784 days 
# 2019 - 8472 days
# 2018 - 8736 days
# 2017 - 8760 days 
# 2016 - 8759 days
# 2015 - 8736 days 
# 2014 - 8712 days 
# 2013 - 8760 days 
# 2012 - 8760 days 
# 2011 - 8736 days
# 2010 - 8640 days 


# In[3]:


# THIS PLOTS THROUGH MANY YEARS OF IRRADIANCE 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from datetime import datetime
cut_off = None

plots = 5
year_begin = 2015

# i can probably make this clearer by using dict or sth
year_dataframes = [] # this to combine the dataframes that the loop iterates through so we can call each element of the dataframe array to get each years data

fig, axes = plt.subplots(plots, 1, figsize=(20, 15),tight_layout=True)

# Iterate through the number of plots (years) that we want
for i in range(0,plots):    
    fileName = "C:/Users/tartn/Documents/Solarize_Data/Coventry/midas-open_uk-radiation-obs_dv-202107_west-midlands_24102_coventry-coundon_qcv-1_2020.csv"
    fileName = fileName.replace('2020',str(year_begin+i))
    print(fileName)
    df = pd.read_csv(fileName, skiprows=list(range(0, 75)))
    # when reading this csv files, switch the '\' to '/' forward slashes
    # since file data frame is uneven, we need to skip the first 0-75 rows to access the real data 
    year = 'data' + str(year_begin+i)
    
    print(year)
    
    year = pd.DataFrame(columns=['Time','Irradiance','Effective Irradiance']) 
    year['Time'] = df['ob_end_time']
    year['Irradiance'] = df['glbl_irad_amt']
    year.drop(df.tail(1).index,inplace=True) # remove the last 'end' value 

    year = year[:cut_off]

    # iterate through the entire'Time' column 
    # subtract the current row with the starting date 
    # get the number of seconds elapsed and divide by 3600 to get hour 
    
    for index, row in year.iterrows():
        year.loc[index,'hours'] = (datetime.strptime(row['Time'],'%Y-%m-%d %H:%M:%S') - datetime.strptime(year.iloc[0]['Time'],'%Y-%m-%d %H:%M:%S')).total_seconds() / 3600
    
    # since irradiance is in KJ/m2 per hour, we need to divide it down to W/m2
    year['Irradiance'] = (year['Irradiance']*1000)/3600
    
    hoursPerMonth = 700
    high = hoursPerMonth
    low = 0
    
    year_dataframes.append(year)
    ax = axes[i]
    axes[i].set_title("Figure"+ str(i))
    
    year.head()

    # Plot 12 months into 1 subplot each time the main year loop iterates 
    for j in range(1, 13): 
        
        sns.lineplot(x=list(range(hoursPerMonth)),y = year['Irradiance'][low:high],label = j, ax=ax)      
        high += hoursPerMonth
        low += hoursPerMonth
        
    ax.set_xlabel('Time (Hours)')
    ax.set_ylabel('Global Solar Irradiance (W/m2)')
    ax.set_title(f'Radiation vs Time ({str(year_begin + i)})')
    ax.legend()
    
    


# In[223]:


# compare JANs 
# change start and end hours to change months 
for i in range(0,5):
    hoursMonthEnd = 700
    hoursMonthBegin = 0
    hoursTotal = hoursMonthEnd - hoursMonthBegin
    sns.lineplot(x=list(range(hoursTotal)),y = year_dataframes[i]['Irradiance'][hoursMonthBegin:hoursMonthEnd],label = 2015+i);

    plt.xlabel('Time (Hours)');
    plt.ylabel('Global Solar Irradiance (W/m2)');
    plt.title('Radiation vs Time (Jan)');
    plt.legend()


# In[222]:


# compare FEBS  
for i in range(0,5):
    hoursMonthEnd = 1400
    hoursMonthBegin = 700
    hoursTotal = hoursMonthEnd - hoursMonthBegin
    sns.lineplot(x=list(range(hoursTotal)),y = year_dataframes[i]['Irradiance'][hoursMonthBegin:hoursMonthEnd],label = 2015+i);

    plt.xlabel('Time (Hours)');
    plt.ylabel('Global Solar Irradiance (W/m2)');
    plt.title('Radiation vs Time (Feb)');
    plt.legend()


# In[4]:


# PLOTS EACH YEAR OVER AND OVER SHOW BIG CURVE 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from datetime import datetime


plots = 5
year_begin = 2015
plt.figure(figsize=(16, 14)) # Figure width, height % putting plt.fig in the loop makes it create a new plot everytime 

# i can probably make this clearer by using dict or sth
year_dataframes = [] # this to combine the dataframes that the loop iterates through so we can call each element of the dataframe array to get each years data


# Iterate through the number of plots (years) that we want
for i in range(0,plots):    
    fileName = "C:/Users/tartn/Documents/Solarize_Data/Coventry/midas-open_uk-radiation-obs_dv-202107_west-midlands_24102_coventry-coundon_qcv-1_2020.csv"
    fileName = fileName.replace('2020',str(year_begin+i))
    print(fileName)
    df = pd.read_csv(fileName, skiprows=list(range(0, 75)))
    # when reading this csv files, switch the '\' to '/' forward slashes
    # since file data frame is uneven, we need to skip the first 0-75 rows to access the real data 
    year = 'data' + str(year_begin+i)
    
    print(year)
    
    year = pd.DataFrame(columns=['Time','Irradiance','Effective Irradiance']) 
    year['Time'] = df['ob_end_time']
    year['Irradiance'] = df['glbl_irad_amt']
    year.drop(df.tail(1).index,inplace=True) # remove the last 'end' value 

    

    # iterate through the entire'Time' column 
    # subtract the current row with the starting date 
    # get the number of seconds elapsed and divide by 3600 to get hour 
    
    for index, row in year.iterrows():
        year.loc[index,'Hours'] = (datetime.strptime(row['Time'],'%Y-%m-%d %H:%M:%S') - datetime.strptime(year.iloc[0]['Time'],'%Y-%m-%d %H:%M:%S')).total_seconds() / 3600
    
    # since irradiance is in KJ/m2 per hour, we need to divide it down to W/m2
    year['Irradiance'] = (year['Irradiance']*1000)/3600
    
    sns.scatterplot(x = year['Hours'],y = year['Irradiance'],data = year,s=8)
    


# In[5]:


# Plot each year Individually to see what the data looks like 

# Notes 
# 2001 unsuable 
# 2002
# 2003
# 2004
# 2007
# 2010
# 2019 - 7/9/2019 - 7/20/2019 there were no rows for these data  

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from datetime import datetime


plots = 20
year_begin = 2001


# i can probably make this clearer by using dict or sth
year_dataframes = [] # this to combine the dataframes that the loop iterates through so we can call each element of the dataframe array to get each years data


# Iterate through the number of plots (years) that we want
for i in range(0,plots):    
    fileName = "C:/Users/tartn/Documents/Solarize_Data/Coventry/midas-open_uk-radiation-obs_dv-202107_west-midlands_24102_coventry-coundon_qcv-1_2020.csv"
    fileName = fileName.replace('2020',str(year_begin+i))
    print(fileName)
    df = pd.read_csv(fileName, skiprows=list(range(0, 75)))
    # when reading this csv files, switch the '\' to '/' forward slashes
    # since file data frame is uneven, we need to skip the first 0-75 rows to access the real data 
    year = 'data' + str(year_begin+i)
    
    print(year)
    
    year = pd.DataFrame(columns=['Time','Irradiance','Effective Irradiance']) 
    year['Time'] = df['ob_end_time']
    year['Irradiance'] = df['glbl_irad_amt']
    year.drop(df.tail(1).index,inplace=True) # remove the last 'end' value 

    

    # iterate through the entire'Time' column 
    # subtract the current row with the starting date 
    # get the number of seconds elapsed and divide by 3600 to get hour 
    
    for index, row in year.iterrows():
        year.loc[index,'Hours'] = (datetime.strptime(row['Time'],'%Y-%m-%d %H:%M:%S') - datetime.strptime(year.iloc[0]['Time'],'%Y-%m-%d %H:%M:%S')).total_seconds() / 3600
    
    # since irradiance is in KJ/m2 per hour, we need to divide it down to W/m2
    year['Irradiance'] = (year['Irradiance']*1000)/3600
    plt.figure(figsize=(16, 14)) # Figure width, height % putting plt.fig in the loop makes it create a new plot everytime 
    plt.title(str(year_begin+i))
    sns.scatterplot(x = year['Hours'],y = year['Irradiance'],data = year,s=10)
    


# In[6]:


# Plot each year Individually to see what the data looks like 
# Plot their average best fit lines too 
# finds all the missing rows too 

# Notes 
# 2001 unsuable 
# 2002
# 2003
# 2004
# 2007
# 2010
# 2019 - 7/9/2019 - 7/20/2019 there were no rows for these data  

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from datetime import datetime


plots = 3
year_begin = 2018
plt.figure(figsize=(15, 10))

# i can probably make this clearer by using dict or sth
year_dataframes = [] # this to combine the dataframes that the loop iterates through so we can call each element of the dataframe array to get each years data

# INITIAL VISUALIZATION
# Iterate through the number of plots (years) that we want
for i in range(0,plots):    
    fileName = "C:/Users/tartn/Documents/Solarize_Data/Coventry/midas-open_uk-radiation-obs_dv-202107_west-midlands_24102_coventry-coundon_qcv-1_2020.csv"
    current_year = year_begin + i
    fileName = fileName.replace('2020',str(current_year))
    print(fileName)
    df = pd.read_csv(fileName, skiprows=list(range(0, 75)))
    # when reading this csv files, switch the '\' to '/' forward slashes
    # since file data frame is uneven, we need to skip the first 0-75 rows to access the real data 
    year = 'data' + str(current_year)
    
    print(year)
    
    year = pd.DataFrame(columns=['Time','Irradiance','Effective Irradiance']) 
    year['Time'] = df['ob_end_time']
    year['Irradiance'] = df['glbl_irad_amt']
    year.drop(df.tail(1).index,inplace=True) # remove the last 'end' value 

    

    # iterate through the entire'Time' column 
    # subtract the current row with the starting date 
    # get the number of seconds elapsed and divide by 3600 to get hour 
    
    for index, row in year.iterrows():
        year.loc[index,'Hours'] = (datetime.strptime(row['Time'],'%Y-%m-%d %H:%M:%S') - datetime.strptime(year.iloc[0]['Time'],'%Y-%m-%d %H:%M:%S')).total_seconds() / 3600
        
        
    
    
    # since irradiance is in KJ/m2 per hour, we need to divide it down to W/m2
    year['Irradiance'] = (year['Irradiance']*1000)/3600
    
    #Plot each graph
    plt.figure(figsize=(16, 14)) # Figure width, height % putting plt.fig in the loop makes it create a new plot everytime 
    plt.title(str(year_begin+i))
    sns.scatterplot(x = year['Hours'],y = year['Irradiance'],data = year,s=16)

    

    
    # DATA PREPROCESSING 
    # Check if the next hour is correct and not skipped 
    # Print all the rows that were skipped 
    missing_rows = []
    for i in range(len(year['Hours'])):
        if i > 0:
            if  (year.iloc[i,3] - year.iloc[i-1,3]) > 1:  #get each row of the 'Hours' Column 
                print('Hour:', end=" ")
                print(i-1),
                print('Skips: ', end=" "),
                print(year.iloc[i,3] - year.iloc[i-1,3] , end=" ")
                print('hours after this ')

                start = int(year.iloc[i-1,3])
                end = int(year.iloc[i,3])

                missing_rows.append(list(range(start,end))) # THESE ARE ALL THE SKIPPED ROWS 

                # for the missing range, get a create new df with the missing hours
            
            
            


    # 12/3/2003  2:00:00 AM % skipped 24th Nov to 3rd Dec

    # Fitting an Average best fit line, instead of top/bottom boundary 
    # Fit a polynomial curve to the data
    # these numbers may seem low but it takes into account the night time as well 
    # I think i can use this code to fill in the data for now 
    total_rows = len(year) 
    hours = year.iloc[0:total_rows,3].values
    irradiance = year.iloc[0:total_rows,1].values

    coefficients = np.polyfit(hours, irradiance,12)
    curve_fit = np.poly1d(coefficients)

    # Calculate corresponding y values for the curve
    y_curve = curve_fit(hours)

    # Draw the curved bounding line
    
    sns.scatterplot(x = hours,y = y_curve,data = year, s = 6)
    
    print(current_year, missing_rows)





# In[5]:


np.nan


# In[50]:


print(missing_rows)


# In[85]:


hours = year.iloc[1000:2000,3].values
irradiance = year.iloc[1000:2000,1].values
np.polyfit(hours,irradiance,2)


# In[52]:


# Plot each year Individually to see what the data looks like 
# Plot their average best fit lines too 
# finds all the missing rows too 

# Notes 
# 2001 unsuable 
# 2002
# 2003
# 2004
# 2007
# 2010
# 2019 - 7/9/2019 - 7/20/2019 there were no rows for these data  

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from datetime import datetime


plots = 15
year_begin = 2005
plt.figure(figsize=(15, 10))

# i can probably make this clearer by using dict or sth
year_dataframes = [] # this to combine the dataframes that the loop iterates through so we can call each element of the dataframe array to get each years data

# INITIAL VISUALIZATION
# Iterate through the number of plots (years) that we want
for i in range(0,plots):    
    fileName = "C:/Users/tartn/Documents/Solarize_Data/Coventry/midas-open_uk-radiation-obs_dv-202107_west-midlands_24102_coventry-coundon_qcv-1_2020.csv"
    current_year = year_begin + i
    fileName = fileName.replace('2020',str(current_year))
    print(fileName)
    df = pd.read_csv(fileName, skiprows=list(range(0, 75)))
    # when reading this csv files, switch the '\' to '/' forward slashes
    # since file data frame is uneven, we need to skip the first 0-75 rows to access the real data 
    year = 'data' + str(current_year)
    
    print(year)
    
    year = pd.DataFrame(columns=['Time','Irradiance','Effective Irradiance']) 
    year['Time'] = df['ob_end_time']
    year['Irradiance'] = df['glbl_irad_amt']
    year.drop(df.tail(1).index,inplace=True) # remove the last 'end' value 

    

    # iterate through the entire'Time' column 
    # subtract the current row with the starting date 
    # get the number of seconds elapsed and divide by 3600 to get hour 
    
    for index, row in year.iterrows():
        year.loc[index,'Hours'] = (datetime.strptime(row['Time'],'%Y-%m-%d %H:%M:%S') - datetime.strptime(year.iloc[0]['Time'],'%Y-%m-%d %H:%M:%S')).total_seconds() / 3600
        
        
    
    
    # since irradiance is in KJ/m2 per hour, we need to divide it down to W/m2
    year['Irradiance'] = (year['Irradiance']*1000)/3600
 
    
    sns.scatterplot(x = year['Hours'],y = year['Irradiance'],data = year,s=1)

    

    
    # DATA PREPROCESSING 
    # Check if the next hour is correct and not skipped 
    # Print all the rows that were skipped 
    missing_rows = []
    for i in range(len(year['Hours'])):
        if i > 0:
            if  (year.iloc[i,3] - year.iloc[i-1,3]) > 1:  #get each row of the 'Hours' Column 
                print('Hour:', end=" ")
                print(i-1),
                print('Skips: ', end=" "),
                print(year.iloc[i,3] - year.iloc[i-1,3] , end=" ")
                print('hours after this ')

                start = int(year.iloc[i-1,3])
                end = int(year.iloc[i,3])

                missing_rows.append(list(range(start,end))) # THESE ARE ALL THE SKIPPED ROWS 

                # for the missing range, get a create new df with the missing hours
            
            
            


    # 12/3/2003  2:00:00 AM % skipped 24th Nov to 3rd Dec

    # Fitting an Average best fit line, instead of top/bottom boundary 
    # Fit a polynomial curve to the data
    # these numbers may seem low but it takes into account the night time as well 
    # I think i can use this code to fill in the data for now 
    total_rows = len(year) 
    hours = year.iloc[0:total_rows,3].values
    irradiance = year.iloc[0:total_rows,1].values

    coefficients = np.polyfit(hours, irradiance,12)
    curve_fit = np.poly1d(coefficients)

    # Calculate corresponding y values for the curve
    y_curve = curve_fit(hours)

    # Draw the curved bounding line
    
    sns.scatterplot(x = hours,y = y_curve,data = year, s = 6)
    
    print(current_year, missing_rows)





# In[55]:


''' 
for each year 
if next hour value isnt previous hour value + 1
insert a row of the next hour's value and insert NaN into the Irradiance column 
and move 1 row down
else move down 1 row 

'''

# Plot each year Individually to see what the data looks like 
# Plot their average best fit lines too 
# finds all the missing rows too and replace into new array 

# Notes 
# 2001 unsuable 
# 2002
# 2003
# 2004
# 2007
# 2010
# 2019 - 7/9/2019 - 7/20/2019 there were no rows for these data  

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from datetime import datetime

# DATFRAME with Irradiance of all years with Index being the hour (contains NaNs to be replaced with averages)
all_years = pd.DataFrame() 

plots = 6
year_begin = 2015
plt.figure(figsize=(15, 10))

# i can probably make this clearer by using dict or sth
year_dataframes = [] # this to combine the dataframes that the loop iterates through so we can call each element of the dataframe array to get each years data

# INITIAL VISUALIZATION
# Iterate through the number of plots (years) that we want
for i in range(0,plots):    
    fileName = "C:/Users/tartn/Documents/Solarize_Data/Coventry/midas-open_uk-radiation-obs_dv-202107_west-midlands_24102_coventry-coundon_qcv-1_2020.csv"
    current_year = year_begin + i
    fileName = fileName.replace('2020',str(current_year))
    print(fileName)
    df = pd.read_csv(fileName, skiprows=list(range(0, 75)))
    # when reading this csv files, switch the '\' to '/' forward slashes
    # since file data frame is uneven, we need to skip the first 0-75 rows to access the real data 
    year = 'data' + str(current_year)
    
    print(year)
    
    year = pd.DataFrame(columns=['Time','Irradiance','Effective Irradiance']) 
    year['Time'] = df['ob_end_time']
    year['Irradiance'] = df['glbl_irad_amt']
    year.drop(df.tail(1).index,inplace=True) # remove the last 'end' value 

    

    # iterate through the entire'Time' column 
    # subtract the current row with the starting date 
    # get the number of seconds elapsed and divide by 3600 to get hour 
    
    for index, row in year.iterrows():
        year.loc[index,'Hours'] = (datetime.strptime(row['Time'],'%Y-%m-%d %H:%M:%S') - datetime.strptime(year.iloc[0]['Time'],'%Y-%m-%d %H:%M:%S')).total_seconds() / 3600
        
        
    
    
    # since irradiance is in KJ/m2 per hour, we need to divide it down to W/m2
    year['Irradiance'] = (year['Irradiance']*1000)/3600
 
    
    sns.scatterplot(x = year['Hours'],y = year['Irradiance'],data = year,s=2)

    

    
    # DATA PREPROCESSING 
    # Check if the next hour is correct and not skipped 
    # Print all the rows that were skipped 
    missing_rows = []
 
    last_hour = year['Hours'].iloc[-1] # 8759 in 2005
    
    
    
    for i in range(len(year['Hours'])):
        if i > 0:
            if  (year.iloc[i,3] - year.iloc[i-1,3]) > 1:  #get each row of the 'Hours' Column 
                print('Hour:', end=" ")
                print(i-1),
                print('Skips: ', end=" "),
                print(year.iloc[i,3] - year.iloc[i-1,3] , end=" ")
                print('hours after this ')

                start = int(year.iloc[i-1,3])
                end = int(year.iloc[i,3])

                missing_rows.append(list(range(start,end))) # THESE ARE ALL THE SKIPPED ROWS 

                # for the missing range, get a create new df with the missing hours
          
        
        
    # Creates a new Dataframe to work on with just Irradiance and Hour
    new_df = pd.concat([year['Irradiance'], year['Hours']], axis=1)
    
    hours_in_year = 8760
    
    # Fills all the missing hours and puts in a NaN for the corresponding Irradiance 
    for h in range(0,hours_in_year):
        if (h > 0) and (h < hours_in_year):
            if (new_df.iloc[h,1] - new_df.iloc[h-1,1]) > 1:
                new_row_df = pd.DataFrame({'Irradiance': [np.nan], 'Hours': [h]}, index=[h])
                new_df = pd.concat([new_df.loc[:h-1], new_row_df, new_df.loc[h:]]).reset_index(drop=True)
                
    # make the new dataFrame with all the irradiances and 8760 hours, then replace them with the averages 
    # if is not NaN sum up and average out for each hour for the 10 years 
    
    all_years[str(current_year)] = new_df['Irradiance']
    

               
        
# After looping for every year to get a new data frame with Irradiances and Nans
# iterate through each year column, then if NaN, replace if Avg of that entire row thats not NaN 

# find the average by counting all non NaNs and dividing by that count for each row 
averages = []
for index1, row1 in all_years.iterrows():
    total = 0
    nums = 0
    for column1, value1 in row1.iteritems():
    
        if pd.isna(value1):   # if no value avaib
            pass
        else: 
            nums += 1
            total += value1
            
    row_average = total/nums
    averages.append(np.round(row_average,2))

all_years['Averages'] = pd.DataFrame(averages)
        
            
        
        

# go through each row of each col, if NaN, replace with the Average value from the same index position 
for col2 in all_years.columns:
    
    for row2, value2 in all_years[col2].iteritems():
        
        if pd.isna(value2):
            
            all_years.loc[row2, col2] = all_years['Averages'][row2]
 


            



    # Fitting an Average best fit line, instead of top/bottom boundary 
    # Fit a polynomial curve to the data
    # these numbers may seem low but it takes into account the night time as well 
    # I think i can use this code to fill in the data for now 
    total_rows = len(year) 
    hours = year.iloc[0:total_rows,3].values
    irradiance = year.iloc[0:total_rows,1].values

    coefficients = np.polyfit(hours , irradiance,12)
    curve_fit = np.poly1d(coefficients)

    # Calculate corresponding y values for the curve
    y_curve = curve_fit(hours)

    # Draw the curved bounding line
    
    sns.scatterplot(x = hours,y = y_curve,data = year, s = 2)
    
    print(current_year, missing_rows)
    
    
plt.figure(figsize=(15, 10))   
 
hours1 = range(len(all_years['Averages'].values))
sns.scatterplot(x = year['Hours'],y = all_years['Averages'],data = year,s=2)
coefficients1 = np.polyfit( hours1, all_years['Averages'].values,6)
curve_fit1 = np.poly1d(coefficients1)

# Calculate corresponding y values for the curve
y_curve1 = curve_fit1(hours1) # fit hours1 into a formula that produces the y values 

# Draw the curved bounding line

sns.scatterplot(x = hours1,y = y_curve1 ,data = all_years, s = 2)
plt.xlabel('Average Solar Irradiance W/m2')
plt.ylabel('Time (days)')


# In[165]:


all_years.head()


# In[52]:


plt.figure(figsize=(10, 8))  
hours1 = range(len(all_years['Averages'].values))
sns.scatterplot(x = year['Hours'],y = all_years['Averages'],data = year,s=2)

coefficients1 = np.polyfit(hours1, all_years['Averages'].values,6)
curve_fit1 = np.poly1d(coefficients1)

# Calculate corresponding y values for the curve
y_curve1 = curve_fit1(hours1)

# Draw the curved bounding line

sns.scatterplot(x = hours1,y = y_curve1 ,data = all_years, s = 2)
plt.xlabel('Average Solar Irradiance W/m2')
plt.ylabel('Time (days)')

