#!/usr/bin/env python
# coding: utf-8

# In[116]:


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


# In[165]:


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


# In[207]:


# THIS PLOTS THROUGH MANY YEARS OF IRRADIANCE 

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
    df = p d.read_csv(fileName, skiprows=list(range(0, 75)))
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


# In[94]:


# Plot of radiation over the entire year 2020

sns.lineplot(x=data['hours'],y = data['Irradiance']);
plt.xlabel('Time (Hours)');
plt.ylabel('Global Solar Irradiance (W/m2)');
plt.title('Radiation vs Time (2020)');

