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





# In[1]:


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
from datetime import *



# DATFRAME with Irradiance of all years with Index being the hour (contains NaNs to be replaced with averages)
all_years = pd.DataFrame() 

plots = 4
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
 


            



##### Fitting an Average best fit line, instead of top/bottom boundary ##################
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







############# SUN ANGLE CALCULATIONS ##########################

from pysolar.solar import *
from datetime import *
from math import *
import matplotlib.pyplot as plt
import calendar


#Coventry Co-ordinates
latitude = 52.424
longitude = -1.536

hours_total = 24
days_total = 29
months_total = 12
year = 2019
month = 6
day = 15
cosRatios = []

for months in range(months_total):
    _, last_day = calendar.monthrange(year, months+1)
    for days in range(1, last_day + 1):
        for hour in range(hours_total):
            date = datetime(year, months+1, days, hour, 13, 1, 130320, tzinfo= timezone.utc)

            #print(hour)

            Sun_altitude = get_altitude(latitude, longitude, date)
            zenith = (90-Sun_altitude) 
            zenith_rad = radians(zenith)
            #print("Zenith deg : %f" % (zenith))

            azimuth = get_azimuth(latitude, longitude, date)
            azimuth_rad = radians(180-azimuth)
            #print("Azimuth deg: %f" % (180 - azimuth)) # since in pysolar the the 0th degree refers to the noth pole which in the formula is in reference to the south pole 
            # therefore we subtract subtract 180 by that angle to get its reference from the south pole 

            altitude_deg = get_altitude(latitude, longitude, date)
            radiation.get_radiation_direct(date, altitude_deg)
            # print("Hours: %d, Radiation : %f" % (hour,radiation.get_radiation_direct(date, altitude_deg)))

            tilt_rad = radians(45)
            surfaceAzimuth_rad = radians(-10)

            # Ratio cos(theta) from the Lamberts formula 
            # Negative value means sun is behind earth can be equated to zero 
            cosTheta = cos(zenith_rad)*cos(tilt_rad)+(sin(zenith_rad)*sin(tilt_rad)*cos(azimuth_rad - surfaceAzimuth_rad))
            #print(cosTheta)
            #print(' ')
            cosRatios.append(cosTheta)

x = range(len(cosRatios))

#plt.figure(figsize=(16, 12))  
#plt.plot(x,cosRatios)
len(cosRatios) # now we can assume these numbers do not change as every year the earth revolves back to the same place 

# replace all negative cosine ratios with zero cuz thats when the sun is behind the earth 

cosRatioDf = pd.DataFrame(cosRatios)
cosRatio = cosRatioDf.applymap(lambda x: max(x, 0))

all_years['cosRatio'] = cosRatio


# FINAL EFFECTIVE IRRADIANCE
# multiply cos()ratios by the recieved average irradiance 
## Numbers may seem low but thats cuz its averaged out for 24 hours, not the 6-10 hours of sunlight we get

all_years['EffectiveAverage'] = all_years['Averages']*all_years['cosRatio']
sns.scatterplot(x=np.arange(0,len(all_years['EffectiveAverage'])),y = all_years['EffectiveAverage'].values, s = 2, marker='s')


coefficients3 = np.polyfit( hours1, all_years['EffectiveAverage'].values,6)
curve_fit3 = np.poly1d(coefficients3)

# Calculate corresponding y values for the curve
y_curve3 = curve_fit3(hours1) # fit hours1 into a formula that produces the y values 

# Draw the curved bounding line

sns.scatterplot(x = hours1,y = y_curve3 ,data = all_years, s = 2)
plt.title('Before and After Considering Effective Angle')



############ Day Average over all years #########################
# Create a new DataFrame for the averaged values
plt.figure(figsize=(15, 10))   
averaged_df = pd.DataFrame()
hoursInDay = 24
daysInYear = 365
combineDays = 24
# Calculate the average for each chunk and append to the new DataFrame
# for each 24 hour blocks of data, average them to get energy produce in WattHours of each day
for i in range(0, len(all_years), hoursInDay):
    block = all_years.iloc[i : i + combineDays]
    averaged_value = block['EffectiveAverage'].mean()
    averaged_df = averaged_df.append({'day_average': averaged_value}, ignore_index=True)
    
len(averaged_df)
sns.scatterplot(x=np.arange(1,daysInYear+1),y=averaged_df['day_average'].values,s=4)
plt.title('All Years Days Average')
plt.xlabel('Days')
plt.ylabel('Average Solar Irradiance (Watt-Hour/m2)')

coefficients4 = np.polyfit(np.arange(1,daysInYear+1),averaged_df['day_average'].values,6)
curve_fit4 = np.poly1d(coefficients4)

# Calculate corresponding y values for the curve
y_curve4 = curve_fit4(np.arange(1,daysInYear+1)) # fit hours1 into a formula that produces the y values 

# Draw the curved bounding line

sns.scatterplot(x = np.arange(1,daysInYear+1),y = y_curve4 ,data = averaged_df, s = 4)


# In[29]:



# Create a new DataFrame for the averaged values
averaged_df = pd.DataFrame()
hoursInDay = 24

# Calculate the average for each chunk and append to the new DataFrame
for i in range(0, len(all_years), hoursInDay):
   block = all_years.iloc[i : i + combineDays]
   averaged_value = block['EffectiveAverage'].mean()
   averaged_df = averaged_df.append({'day_average': averaged_value}, ignore_index=True)
   
len(averaged_df)
sns.scatterplot(x=np.arange(1,366),y=averaged_df['day_average'].values,s=5)
plt.title('All Years Days Average')
plt.xlabel('Days')
plt.ylabel('Average Solar Irradiance (W/m2)')


# In[34]:


len(all_years['EffectiveAverage'])


# In[58]:


############ Day Average over all years #########################
# Create a new DataFrame for the averaged values
plt.figure(figsize=(15, 10))   
averaged_df = pd.DataFrame()
hoursInDay = 24
daysInYear = 365
# Calculate the average for each chunk and append to the new DataFrame
# for each 24 hour blocks of data, average them to get energy produce in WattHours of each day
for i in range(0, len(all_years), hoursInDay):
    block = all_years.iloc[i : i + combineDays]
    averaged_value = block['EffectiveAverage'].mean()
    averaged_df = averaged_df.append({'day_average': averaged_value}, ignore_index=True)
    
len(averaged_df)
sns.scatterplot(x=np.arange(1,daysInYear+1),y=averaged_df['day_average'].values,s=4)
plt.title('All Years Days Average')
plt.xlabel('Days')
plt.ylabel('Average Solar Irradiance (Watt-Hour/m2)')

coefficients4 = np.polyfit( np.arange(1,daysInYear+1),averaged_df['day_average'].values,4)
curve_fit4 = np.poly1d(coefficients4)

# Calculate corresponding y values for the curve
y_curve4 = curve_fit4(np.arange(1,daysInYear+1)) # fit hours1 into a formula that produces the y values 

# Draw the curved bounding line

sns.scatterplot(x = np.arange(1,daysInYear+1),y = y_curve4 ,data = averaged_df, s = 2)


# In[240]:


# Code to get Clear Sky Radiation per hour 
# To check against the MET Office Data 


from pysolar.solar import *
from datetime import *
from math import *
import matplotlib.pyplot as plt
import calendar


#Coventry Co-ordinates
latitude = 52.424
longitude = -1.536

hours_total = 24
days_total = 29
months_total = 12
year = 2019
month = 6
day = 15
cosRatios = []

for months in range(months_total):
    _, last_day = calendar.monthrange(year, months+1)
    for days in range(1, last_day + 1):
        for hour in range(hours_total):
            date = datetime(year, months+1, days, hour, 13, 1, 130320, tzinfo= timezone.utc)

            #print(hour)

            Sun_altitude = get_altitude(latitude, longitude, date)
            zenith = (90-Sun_altitude) 
            zenith_rad = radians(zenith)
            #print("Zenith deg : %f" % (zenith))

            azimuth = get_azimuth(latitude, longitude, date)
            azimuth_rad = radians(180-azimuth)
            #print("Azimuth deg: %f" % (180 - azimuth)) # since in pysolar the the 0th degree refers to the noth pole which in the formula is in reference to the south pole 
            # therefore we subtract subtract 180 by that angle to get its reference from the south pole 

            altitude_deg = get_altitude(latitude, longitude, date)
            radiation.get_radiation_direct(date, altitude_deg)
            # print("Hours: %d, Radiation : %f" % (hour,radiation.get_radiation_direct(date, altitude_deg)))

            tilt_rad = radians(45)
            surfaceAzimuth_rad = radians(-10)

            # Ratio cos(theta) from the Lamberts formula 
            # Negative value means sun is behind earth can be equated to zero 
             
            #print(cosTheta)
            #print(' ')
            cosRatios.append(cosTheta)

x = range(len(cosRatios))

#plt.figure(figsize=(16, 12))  
#plt.plot(x,cosRatios)
len(cosRatios) # now we can assume these numbers do not change as every year the earth revolves back to the same place 

# replace all negative cosine ratios with zero cuz thats when the sun is behind the earth 

cosRatioDf = pd.DataFrame(cosRatios)
cosRatio = cosRatioDf.applymap(lambda x: max(x, 0))


# In[87]:


# Trying to find optimal 
from sympy import symbols, Eq, cos, sin, nsolve, deg

# Define the variables
t, s = symbols('t s') # tilt and surface azimuth
z = 120 # zenith 
g = 120 # azimuth

# Define the system of gradient equations
equation1 = Eq(-cos(z)*sin(t) + sin(z)*cos(t)*cos(s-g), 0)
equation2 = Eq(-sin(z)*sin(t)*sin(s-g), 0)

# Solve the system numerically
numerical_solution = nsolve([equation1, equation2], (t, s), (0, 0))

# Convert the numerical solutions to degrees
numerical_solution_deg = (deg(numerical_solution[0]), deg(numerical_solution[1]))

# Print the numerical solution in degrees
print("Numerical Solution (Degrees):", numerical_solution_deg)


# In[320]:


import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate some sample data
x = np.array(range(1, 366))
y = averaged_df['day_average'].values

# Reshape x to a 2D array (required for scikit-learn)
x_reshaped = x.reshape(-1, 1)

# Create polynomial features 
poly = PolynomialFeatures(degree=6)
X_poly = poly.fit_transform(x_reshaped)

# Split the data into training and test sets (removed scaling for now)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=None)

# Solve linear regression using numpy.linalg.lstsq
weights = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

# Predict on the test data
y_pred_test = np.dot(X_test, weights)

# Calculate mean squared error on the test data
mse_test = mean_squared_error(y_test, y_pred_test)

# Calculate root mean squared error (RMSE)
rmse_test = np.sqrt(mse_test)

print(f'Root Mean Squared Error on Test Data: {rmse_test}')

# Plot the data and the fitted curve
x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
y_pred = np.dot(x_range_poly, weights)

plt.scatter(x, y, label='Test Data', s=5)  # Set the size with the 's' parameter
plt.plot(x_range, y_pred, color='red', label='Fitted Curve (Linear Regression)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
print(weights)


# In[189]:


# Split RIDGE INTO Random TRAIN AND TEST AND MSE ERROR shown
import numpy as np
from math import *
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Generate some sample data
x1 = np.array(range(1, 366))
y1 = averaged_df['day_average'].values

# Reshape x to a 2D array (required for scikit-learn)
x_reshaped = x.reshape(-1, 1)

# Create polynomial features (degree=6 for a 6th-degree polynomial)
poly = PolynomialFeatures(degree=6)
X_poly = poly.fit_transform(x_reshaped)

# Scale features
scaler = StandardScaler() # standardize feature by removing mean and scale to unit variance
X_poly_scaled = scaler.fit_transform(X_poly)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_poly_scaled, y1, test_size=0.2, random_state=None)

# Increase regularization strength % 0.0005 gives great results 
alpha = 0.0005


# Create and fit the Ridge Regression model on the training data
ridge_model = Ridge(alpha=alpha)
ridge_model.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred_test = ridge_model.predict(X_test)

# Plot the data and the fitted curve
x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
x_range_poly = poly.transform(x_range)
x_range_scaled = scaler.transform(x_range_poly)
y_pred = ridge_model.predict(x_range_scaled)

plt.scatter(x1, y1, label='Test Data', s=5)  # Set the size with the 's' parameter
plt.plot(x_range, y_pred, color='red', label=f'Fitted Curve (Ridge Regression, alpha={alpha})')
plt.xlabel('day')
plt.ylabel('Energy (Watt-Hour)')
plt.legend()
plt.show()
mse_test = mean_squared_error(y_test, y_pred_test)
error = np.sqrt(mse_test)
print(error)
print(ridge_model.coef_)

# test and train in diff colors 
# plot predicted vs true data should be straight line


# In[329]:


ridge_model.predict([[1]+[0]*6])


# In[202]:


# SCALE THIS BY THE AVERAGE TEMP RISE IN UK / YEAR?? for multiplying each year??
# we can take the average between every 2 rows of data as the daily mean then calculate using the formula the multiplier efficiency of the panel 
# only a few years are available 
# some years dont have all the data 

fileName = "C:/Users/tartn/Documents/Solarize_Data/CoventryTemperature/midas-open_uk-daily-temperature-obs_dv-202308_west-midlands_24102_coventry-coundon_qcv-1_2017.csv"

temps = pd.read_csv(fileName, skiprows=list(range(0, 90)))


temperatures = pd.DataFrame()
temperatures['time'] = temps.iloc[:,0]
temperatures['max'] = temps.iloc[:,8]
temperatures['min'] = temps.iloc[:,9]


tempAvgDouble = []
for itemp in range(0,len(temperatures['time'])):
    tempAvgDouble.append((temperatures['max'][itemp]+ temperatures['min'][itemp])/2)

temperatures['averages'] = tempAvgDouble

temperatures = temperatures.iloc[:-1] # remove the last NaN row
temperatures.head()

dayTemps = []
# finding average temp every 9am=9pm (i think this is the best i can do cuz thats the only available data and night time 9pm-9am is mostly irrelevant)
for jtemp in range(0,floor(len(temperatures['time'])/2)):
    
    dayTemps.append(temperatures['averages'][2*jtemp+1])
    
tempsAvg = pd.DataFrame()
tempsAvg['temperatures'] = dayTemps



# Set parameters from panel model 
initialEff = 21.8*0.985
degradeRate = 0.25
TC = -0.29
standardTestTemp = 25

# these will change as time progresses 
yearCount = 5
currentTemp = 5 # for row in year 

#EfficiencyFormula =initialEff + TC*(currentTemp-standardTestTemp) - (yearCount-1)*(degradeRate)
print(efficiency)
efficiency = []

tempsAvg['finalEfficiency'] = np.round(initialEff + TC*(tempsAvg['temperatures']-standardTestTemp) - (yearCount-1)*(degradeRate),2)/100

totalEnergy = pd.DataFrame()
totalEnergy['energy'] = averaged_df['day_average']*tempsAvg['finalEfficiency'] 


x = np.array(range(1, 366))
y = totalEnergy['energy'].values
plt.scatter(x, y,s=5)
plt.xlabel('day')
plt.ylabel('Effective Energy (WH) per m^2')
plt.title('Energy recieved VS Panel Output')

# NOTE that this uses the average irradiance and ridge regression and temperature data from 2017


### this is for comparing with ridge, run code in prev block first!
plt.scatter(x1, y1, label='Test Data', s=5)  # Set the size with the 's' parameter
plt.plot(x_range, y_pred, color='red', label=f'Fitted Curve (Ridge Regression, alpha={alpha})')

###

