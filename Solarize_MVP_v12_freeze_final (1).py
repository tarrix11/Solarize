#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Final code 
# By Nithit Kongsuphol u2055367


# In[2]:


# NUMBER 1

''' 
THIS IS THE MAIN CODE 
gets effective irradiance - still gotta find optimal angles 
'''

# Plot each year Individually to see what the data looks like 
# Plot their average best fit lines too 
# finds all the missing rows too and replace into new array 


import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from datetime import *


# DATFRAME with Irradiance of all years with Index being the hour (contains NaNs to be replaced with averages)
all_years = pd.DataFrame() 

plots = 9
year_begin = 2010
plt.figure(figsize=(15, 10))


year_dataframes = [] # this to combine the dataframes that the loop iterates through so we can call each element of the dataframe array to get each years data

# INITIAL VISUALIZATION
# Iterate through the number of plots (years) that we want
for i in range(0,plots):    
    fileName = "C:/Users/tartn/Documents/Solarize_Data/Kew_Gardens/midas-open_uk-radiation-obs_dv-202107_greater-london_00723_kew-gardens_qcv-1_2020.csv"
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


############# SUN ANGLE CALCULATIONS #############

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


# In[3]:


# NUMBER 2

# Split RIDGE INTO Random TRAIN AND TEST AND MSE ERROR shown
# LASSO didnt perform well at all
#Now using Bayesian Ridge Instead

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

days = np.array(range(1, 366))
avgIrradiance = averaged_df['day_average'].values
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_positions = np.linspace(0, 365, num=12)

# Reshape og data to an 2D array (sklearn wants this format)
days_reshaped = days.reshape(-1, 1) # this just makes it [[array]] instead of [array]

# Create polynomial features with a degree of 6
fit_curve = PolynomialFeatures(degree= 6)
fitted_curve = fit_curve.fit_transform(days_reshaped)
# Scale those features so that they are standardized 
scaler = StandardScaler() # standardize feature by removing mean and scale to unit variance
fitted_curve_scaled = scaler.fit_transform(fitted_curve)

# Split data into training and test sets randomly 
X_train, X_test, y_train, y_test = train_test_split(fitted_curve_scaled, avgIrradiance, test_size=0.2, random_state=None)

# Fit the sets into bayesian ridge regression to predict
bayesian_model = BayesianRidge()
bayesian_model.fit(X_train, y_train)

# This is just incase I need to measure performance of the mode on unseen data, in this case its just modelling known data
# Evaluate the model on test data
y_pred_test, y_std_test = bayesian_model.predict(X_test, return_std=True)

# Plot the data and the fitted curve
x_range = np.linspace(min(days), max(days), 365).reshape(-1, 1)
x_range_poly = fit_curve.transform(x_range)
x_range_scaled = scaler.transform(x_range_poly)
y_pred, y_std = bayesian_model.predict(x_range_scaled, return_std=True)
plt.scatter(days, avgIrradiance, label='Averaged Irradiance', s=5)  # Set the size with the 's' parameter
plt.plot(x_range, y_pred, color='red', label='Fitted Bayesian Ridge Regression')
plt.fill_between(x_range.flatten(), y_pred - y_std, y_pred + y_std, color='red', alpha=0.2)
plt.xlabel('Days of the year')
plt.ylabel('Energy (Watt-Hour)')
plt.legend()
plt.xticks(month_positions, months)
plt.title('Bayesian Ridge Regression on Averaged Solar Irradiance')
plt.savefig("C:/Users/tartn/Pictures/solarize/BayesianRidgeIrradiance.png", dpi=400)
plt.show()

mse_test = mean_squared_error(y_test, y_pred_test)
error = np.sqrt(mse_test)
#print(error) # RMSE error avg 
#print(bayesian_model.coef_)
#print(y_std) # use this to plus minus the thing 
irradiance_abs_uncertainity = y_std


# In[4]:


# Number 3

# SCALE THIS BY THE AVERAGE TEMP RISE IN UK / YEAR for multiplying each year
# only a few years are available 
# some years dont have all the data 

# RUN MAIN CODE FIRST 
current_year = 2000
plots = 20
for i in range(0,plots):
    fileNameT = "C:/Users/tartn/Documents/Solarize_Data/KewTemperatures/midas-open_uk-daily-temperature-obs_dv-202308_greater-london_00723_kew-gardens_qcv-1_2017.csv"
    fileNameT = fileNameT.replace('2020',str(current_year))
    temps = pd.read_csv(fileNameT, skiprows=list(range(0, 90)))

    # extract min and max of each time 
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
    colT = 'temperatures' + str(current_year)
    tempsAvg[colT] = dayTemps

    daysInYear = 365
    tempsAvg = tempsAvg.head(daysInYear)


    # Set parameters from panel model 
    initialEff = 21.8*0.985
    degradeRate = 0.25
    TC = -0.29
    standardTestTemp = 25

    # these will change as time progresses 
    yearCount = 5
    currentTemp = 5 # for row in year 

    #EfficiencyFormula =initialEff + TC*(currentTemp-standardTestTemp) - (yearCount-1)*(degradeRate)

    efficiency = []
    averaged_df = averaged_df.head(daysInYear)
    averaged_irradiance = averaged_df

    tempsAvg['finalEfficiency'] = np.round(initialEff + TC*(tempsAvg[colT]-standardTestTemp) - (yearCount-1)*(degradeRate),2)/100

    totalEnergy = pd.DataFrame()
    totalEnergy['energy'] = averaged_df['day_average']*tempsAvg['finalEfficiency'] 


    x = np.array(range(0, len(tempsAvg[colT].values)))
    y = totalEnergy['energy'].values
    plt.scatter(x, y,s=5,label='Output')
    plt.xlabel('day')
    plt.ylabel('Effective Energy (WH) per m^2')
    plt.title('Energy recieved VS Panel Output Energy')

    ### this is for comparing with ridge, RUN RIDGE CODE in prev block first!
    plt.scatter(days, avgIrradiance, label='Test Data', s=5)  # Set the size with the 's' parameter
    plt.plot(x_range, y_pred, color='red', label=f'Fitted Curve (Ridge Regression, alpha={0.2})')

    ###
    # Add a legend
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    
    current_year += 1
    print(current_year)


# In[5]:


# Number 4
# effective output with temperature data but not too good cuz missing data

# RUN NUMBER 1 & 2 First 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import floor

# make sure main code is initialized then Ridge code before running this

start_year = 2001
plots = 20

# Set parameters from panel model
initialEff = 21.8 * 0.985
degradeRate = 0.25
TC = -0.29
standardTestTemp = 25

# Create subplots
fig, axes = plt.subplots(plots, 1, figsize=(10, 6 * plots))

# TEMPS DAY AVG OF ALL YEARS
all_temps = pd.DataFrame()

for i in range(plots):
    current_year = start_year + i  # Increment the year for each iteration
    fileNameT = "C:/Users/tartn/Documents/Solarize_Data/KewTemperatures/midas-open_uk-daily-temperature-obs_dv-202308_greater-london_00723_kew-gardens_qcv-1_2020.csv"
    fileNameT = fileNameT.replace('2020', str(current_year))
    temps = pd.read_csv(fileNameT, skiprows=list(range(0, 90)))

    # Extract min and max of each timestep
    temperatures = pd.DataFrame()
    temperatures['time'] = temps.iloc[:, 0]
    temperatures['max'] = temps.iloc[:, 8]
    temperatures['min'] = temps.iloc[:, 9]

    tempAvgDouble = []
    for itemp in range(0, len(temperatures['time'])):
        tempAvgDouble.append((temperatures['max'][itemp] + temperatures['min'][itemp]) / 2)

    temperatures['averages'] = tempAvgDouble

    # Interpolate missing values
    # WRITE THE CODE to convert time to intervals 
    

    dayTemps = []
    # Finding average temp every 9am-9pm
    # only appends the day and not the night so skip half of it
    for jtemp in range(0, floor(len(temperatures['time']) / 2)): 
        dayTemps.append(temperatures['averages'][2 * jtemp + 1])

    tempsAvg = pd.DataFrame()
    colT = 'temperatures' + str(current_year)
    
    # Makes sure column size uniformity, replace missing values with NaN 
    for days1 in range(daysInYear - len(dayTemps)):
        dayTemps.append(np.nan)
    
    tempsAvg[colT] = dayTemps
    
    daysInYear = 365 # doesnt take leap years for uniformity
    tempsAvg = tempsAvg.head(daysInYear)

    #fill in the temosAvg col to be 365 

    all_temps[colT] = tempsAvg[colT]

    # These will change as time progresses
    yearCount = i

    # TEMPERATURE COEFFICIENT FORMULA TO FIND EFFECTIVE EFFICIENCY
    tempsAvg['finalEfficiency'] = np.round(
        initialEff + TC * (tempsAvg[colT] - standardTestTemp) - (i) * (degradeRate), 2) / 100

    totalEnergy = pd.DataFrame()
    totalEnergy['energy'] = averaged_df['day_average'] * tempsAvg['finalEfficiency']

    # Plotting on the i-th subplot
    ax = axes[i]

    x = np.array(range(0, len(tempsAvg[colT].values)))
    y = totalEnergy['energy'].values[:len(x)]  # Ensure y has the same length as x
    ax.scatter(x, y, s=5, label=' Panel Energy Output')
    ax.set_xlabel('day')
    ax.set_ylabel('Effective Energy (WH) per m^2')
    ax.set_title(f'Energy received VS Panel Output Energy - {current_year}')

    # NOTE that this uses the average irradiance and ridge regression (AVERAGE IRRADIANCE )
    ax.scatter(days, avgIrradiance, label='Test Data', s=5)  # Set the size with the 's' parameter
    ax.plot(x_range, y_pred, color='red', label=f'Fitted Total Average Irradiance Curve (Ridge Regression, alpha={0.2})')

    ax.legend()

    print(current_year)
    print(fileNameT)

plt.tight_layout()
plt.show()

# cant see the output in some cuz the temperature data isnt there 

# NOW PUT THE FIXED DATA INTO THE PLOTS!!! -> Label each df properly and see what went wrong maybe sections of code redefines


# In[6]:


# NUMBER 5
# Ridge regression of temperature toreduce noise and fill in missing data gaps

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import re

# df of all the temps of the decades
all_temps_new = all_temps.drop(all_temps.columns[:9], axis=1)

# Replace NaN values with the average of 2 days before and 2 days after
all_temps_new = all_temps_new.interpolate(method='linear', limit=2, limit_direction='both')

# iterate through the cols and make a ridge regression to model the trend 
models = {}
fitted_temps = pd.DataFrame()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_positions = np.linspace(0, 365, num=12)

for year in all_temps_new.columns:

    X = np.array(range(len(all_temps_new))).reshape(-1, 1)  # Day of the year
    y = all_temps_new[year].values  # Temperature

    # Remove rows with NaN values(masking technique)
    show = ~np.isnan(y)
    X_fit = X[show]
    y_fit = y[show]
  
    # 4th degree is the least that can still fit the trend 
    poly = PolynomialFeatures(degree=4)
    X_poly_fit = poly.fit_transform(X_fit)

    # Create and fit the model
    alpha_value = 10
    
    model = Ridge(alpha=alpha_value)
    model.fit(X_poly_fit, y_fit)
    
    models[year] = model
    print(f'Alpha for {year}: {alpha_value}')
    models[year] = model

    # Plot the original data
    plt.scatter(X, y, color='blue',s=1, label = 'Temperature Data')

    # Create predictions for the entire year
    X_poly = poly.transform(X)
    y_pred = model.predict(X_poly)

    # PLOT THE PREDICTION
    plt.plot(X, y_pred, color='red', label = 'Ridge Regression')
    
    year_number = re.findall('\d+', year)[0]
    plt.title(f'Ridge Regression of Temperatures in {year_number}')
    plt.ylabel('Temperature (Celcius)')
    plt.xticks(month_positions, months)
    plt.legend()
    
    # this is just to save the pictures 
    if year_number == '2018':
        plt.savefig("C:/Users/tartn/Pictures/solarize/TempsRidgePrediction2018.png", dpi=400, bbox_inches='tight')
    elif year_number == '2013':
        plt.savefig("C:/Users/tartn/Pictures/solarize/TempsRidgePrediction2013.png", dpi=400, bbox_inches='tight')
    
    plt.show()

    fitted_temps[year] = pd.Series(y_pred)

# Ensure the new DataFrame has the same index as the original DataFrame
fitted_temps.index = all_temps_new.index


# In[7]:


# Number 6
# Bayesian Ridge Regression for each day of the 25 projected years, with 1sd boundaries 

import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

temps_projection_all = pd.DataFrame()
uncertainties_all = pd.DataFrame()
x = np.linspace(0, 34, 100)  # x values from 0 to 34
xticks = np.arange(0, 40, 5)  # x values from 0 to 34
xticklabels = np.arange(2010, 2050, 5)  # Corresponding labels from 

for index, row in fitted_temps.iterrows():
    year1 = re.findall('\\d+', year)
    date = datetime(int(year1[0]), 1, 1) + timedelta(int(index) - 1)
    month_name = date.strftime('%B')
    day_of_month = date.day
    
    X = np.array(range(len(row))).reshape(-1, 1)
    y = row.values.reshape(-1, 1)

    # Standardize both variables of time and temp
    scaler_Years = StandardScaler()
    scaler_Temps = StandardScaler()
    
    X_std = scaler_Years.fit_transform(X)
    y_std = scaler_Temps.fit_transform(y)

    model = BayesianRidge()
    model.fit(X_std, y_std.ravel())

    # create the prediction
    y_pred_std, y_std_std = model.predict(X_std, return_std=True)

    # Unstandardize the predictions and uncertainties
    y_pred = scaler_Temps.inverse_transform(y_pred_std.reshape(-1, 1))
    y_std = y_std_std * scaler_Temps.scale_

    # Generate predictions for 20 future data points
    X_future = np.array(range(len(row)-1, len(row) + 25+1)).reshape(-1, 1)
    X_future_std = scaler_Years.transform(X_future)
    y_future_std, y_future_std_std = model.predict(X_future_std, return_std=True)

    # Unstandardize back to normal
    y_future = scaler_Temps.inverse_transform(y_future_std.reshape(-1, 1))
    y_future_std = y_future_std_std * scaler_Temps.scale_

    # Add the future predictions & uncertainty to the new DataFrame
    # gotta flatten it to make it 1D again
    temps_projection_all = temps_projection_all.append(pd.Series(y_future.flatten(), name=index))
    uncertainties_all = uncertainties_all.append(pd.Series(y_future_std.flatten(), name=index))
    
    plt.figure()
    plt.scatter(X, y, color='black')
    plt.plot(X, y_pred, color='blue', linewidth=3, label='Fitted Data')  # Add label
    plt.fill_between(X.flatten(), y_pred.flatten() - y_std, y_pred.flatten() + y_std, color='blue', alpha=0.2)
    plt.plot(X_future, y_future, color='red', linewidth=3, label='Future Predictions')  # Add label
    plt.fill_between(X_future.flatten(), y_future.flatten() - y_future_std, y_future.flatten() + y_future_std, color='red', alpha=0.2)
    plt.title(f'Bayesian Ridge Regression of Temperature on: {day_of_month}/{month_name}')

    plt.xticks(xticks, xticklabels, rotation=45)
    plt.xlabel('Years')
    plt.ylabel('Temperature (Celcius)')
    plt.legend()  
    
    if index == 168:
         plt.savefig("C:/Users/tartn/Pictures/solarize/BayesianRidgeTempDay168.png", dpi=400, bbox_inches='tight')
    elif index == 40:
        plt.savefig("C:/Users/tartn/Pictures/solarize/BayesianRidgeTempDay14.png", dpi=400, bbox_inches='tight')

    plt.show()

# Print the new DataFrame with the future predictions
temps_projection_all.columns = [str(2020 + int(col)) for col in temps_projection_all.columns]

plt.figure(figsize=(15, 6))

# make colormap for figure 
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, len(temps_projection_all.columns)))

# Loop through each column in the DataFrame
for i, column in enumerate(temps_projection_all.columns):
    plt.plot(temps_projection_all.index, temps_projection_all[column], label=column, color=colors[i])

plt.xlabel('Days of the Year')
plt.ylabel('Temperature (Celcius)')
plt.title('Temperature throughout the years')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()

# Calculate the mean and standard deviation of the temperatures and uncertainties
mean_temps = temps_projection_all.mean(axis=1)
std_temps = temps_projection_all.std(axis=1)
mean_uncertainties = uncertainties_all.mean(axis=1)
std_uncertainties = uncertainties_all.std(axis=1)

plt.figure(figsize=(15, 6))
# Ploting the mean temperature and uncertainty
plt.plot(mean_temps.index, mean_temps, label='Mean Temperature', color='blue')
plt.fill_between(mean_temps.index, (mean_temps - std_temps), (mean_temps + std_temps), color='blue', alpha=0.2)
plt.plot(mean_uncertainties.index, mean_uncertainties, label='Mean Uncertainty', color='red')
plt.fill_between(mean_uncertainties.index, (mean_uncertainties - std_uncertainties), (mean_uncertainties + std_uncertainties), color='red', alpha=0.2)

plt.xlabel('Days of the year')
plt.ylabel('Temperature (Celcius)')
plt.title('Temperature uncertainty throughout the years (1sd)')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

plt.show()


# In[8]:



import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta


day = 167
year = 2045 

date = datetime(year, 1, 1) + timedelta(day - 1)
month_name = date.strftime('%B')
day_of_month = date.day

# mean and standard deviation of the temperature projection for that day and that year
mean = temps_projection_all.iloc[day-1, year-2024]
std = uncertainties_all.iloc[day-1, year-2024]

# make normal distribution (obj) with the mean and standard deviation
dist = norm(loc=mean, scale=std)

# Create a range of values for the x-axis (temperature)
# I chose 4sd so it covers almost 100%
x = np.linspace(mean - 4*std, mean + 4*std, 100)


# Calculate the probability density function values for the x-axis
y = dist.pdf(x)

plt.plot(x, y, linewidth=5, label='Distribution')

# Fill between mean and 1 standard deviation
plt.fill_between(x, y, where=((x >= mean - std) & (x <= mean + std)), color='g', alpha=0.5, label='1 SD')

# Fill between 1 standard deviation and 2 standard deviations
plt.fill_between(x, y, where=((x >= mean - 2*std) & (x <= mean - std)) | ((x >= mean + std) & (x <= mean + 2*std)), color='b', alpha=0.5, label='2 SD')

# Fill between 2 standard deviations and 3 standard deviations
plt.fill_between(x, y, where=((x >= mean - 3*std) & (x <= mean - 2*std)) | ((x >= mean + 2*std) & (x <= mean + 3*std)), color='r', alpha=0.5, label='3 SD')

# Add the legend to the plot
plt.legend(loc='upper right')

plt.ylabel('Normalise porbability')
plt.xlabel('Temperature (Celcius)')
plt.title('Probability Distribution of Temperature on 16/June/2045')
plt.savefig("C:/Users/tartn/Pictures/solarize/Temp_distribution.png", dpi=400, bbox_inches='tight')
plt.show()


# In[9]:


# Number 7
# Surface plot to illustrate the temperature increase over the years  


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# make mesh grid like for a surf 
X, Y = np.meshgrid(np.arange(temps_projection_all.shape[1]), temps_projection_all.index)

months = ['Jan', 'Apr', 'Jul', 'Oct']
month_positions = np.linspace(0, 365, num=4)


# Convert the DataFrame to a matrix to get Z coordinates
Z = temps_projection_all.values

fig = plt.figure(figsize=(10, 10))  # Increase the size of the figure
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)  # Add a color gradient

years = np.arange(2020, 2046, 5)
plt.xticks(np.arange(0, len(years)*5, 5), years, rotation=0)  # Angle x ticks
ax.set_xlabel('Years')

ax.set_ylabel('Months')
plt.yticks(month_positions, months, rotation=0)  # Angle y ticks
ax.set_xlabel('Years')
ax.set_ylabel('Months')

ax.set_zlabel('Temperature (Celsius)')
ax.set_title('Temperature Throughout the Years')

fig.colorbar(surf,shrink=0.5)  # Add a color bar which maps values to colors.


plt.savefig("C:/Users/tartn/Pictures/solarize/temps_all.png", dpi=400, bbox_inches='tight')
plt.show()


# In[10]:


# NUMBER 8

# Final Efficiency over the years (if temperature increases)

import pandas as pd
import numpy as np
import matplotlib.cm as cm


df_efficiency = pd.DataFrame()
months = ['Jan', 'Apr', 'Jul', 'Oct']
month_positions = np.linspace(0, 365, num=4)

# Iterate over the columns in the original dataframe
for iteration, year in enumerate(temps_projection_all.columns):
    # Get the temperature data for the year
    year_temps = temps_projection_all[year]
 
    
    # Calculate the efficiency for each day and store it in the new dataframe
    df_efficiency[year] = np.round(initialEff + TC * (year_temps - standardTestTemp) - (iteration) * (degradeRate), 2) / 100

                                    #initialEff + TC * (np.array(fitted_temps.iloc[:,-1]) - standardTestTemp) - (iteration) * (degradeRate), 2) / 100
# Now df_efficiency contains the efficiency data

fig = plt.figure(figsize=(10, 9))

cmap = cm.get_cmap('viridis')  

# iterate over years to get the efficiency that we have calcualted
for iteration, year in enumerate(df_efficiency.columns):
    # Only plot for every 5th year
    if int(year) % 5 == 0:
        # Get the color for the line based on its position
        color = cmap(float(iteration) / len(df_efficiency.columns))
        plt.plot(df_efficiency[year], color=color, label=year, linewidth= 3)

    
    
    
    
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=14)
plt.xticks(month_positions, months, fontsize=16)
plt.ylabel('Solar panel efficiency (%)', fontsize=16)
plt.title('Efficiency Throughout the Years (Warming Factored)', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.savefig("C:/Users/tartn/Pictures/solarize/efficinecy_temps_change.png", dpi=400, bbox_inches='tight')
plt.show()


irradiance_abs_uncertainity_df = pd.DataFrame(irradiance_abs_uncertainity)
temps_abs_uncertainity = uncertainties_all 
# Multiply the efficiencies by each other their unertainities to get their absolute uncertainties 
efficiency_relative_uncertainity_change = temps_abs_uncertainity/df_efficiency
irradiance_relative_uncertainty = pd.DataFrame()

temps_abs_uncertainity.columns = [str(int(col) + 2020) for col in temps_abs_uncertainity.columns]
temperature_relative_uncertainity = temps_abs_uncertainity/temps_projection_all
efficiency_relative_uncertainity_change = temperature_relative_uncertainity # since the other factors assumed no uncertainty 
irradiance_relative_uncertainty = irradiance_abs_uncertainity_df.values/averaged_irradiance.values


# In[11]:


# Number 9 
# Calculate the combined uncertainities

# Hourly watt-hour uncertainity for climate change temps
import pandas as pd
import numpy as np

# FOR CHANGING TEMP Root(sum(relative uncertainties))
# Initialize an empty DataFrame
annual_output_uncertainty_change = pd.DataFrame()
annual_output_uncertainty_const = pd.DataFrame()


# For each column in the DataFrame
for i, column in enumerate(efficiency_relative_uncertainity_change.columns):
    # Square the values in the column
    efficiency_squared = efficiency_relative_uncertainity_change[column] ** 2

    # Square the values in the corresponding numpy array
    irradiance_squared = (irradiance_relative_uncertainty ** 2).flatten()

    # Add the squared values
    sum_of_squares = efficiency_squared.values + irradiance_squared

    # Take the square root of the sum of squares
    yearly = pd.DataFrame(np.sqrt(sum_of_squares))*averaged_irradiance.values
    
    # Set the column name to be the iteration number
    yearly.columns = [str(i)]
    

    # Append the yearly DataFrame to the annual_output_uncertainty_change DataFrame
    annual_output_uncertainty_change = pd.concat([annual_output_uncertainty_change, yearly], axis=1)
    print(yearly)
    #annual_output_uncertainty_change = annual_output_uncertainty_change.iloc[:, :-1]
    
    
# FOR CONSTANT TEMP Root(sum(relative uncertainties))
# For each column in the DataFrame
for i, column in enumerate(efficiency_relative_uncertainity_change.columns):
    # Square the values in the column
    efficiency_squared = efficiency_relative_uncertainity_change.iloc[:,0] ** 2

    # Square the values in the corresponding numpy array
    irradiance_squared = (irradiance_relative_uncertainty ** 2).flatten()

    # Add the squared values
    sum_of_squares = efficiency_squared.values + irradiance_squared

    # Take the square root of the sum of squares
    yearly = pd.DataFrame(np.sqrt(sum_of_squares))*averaged_irradiance.values

    # Set the column name to be the iteration number
    yearly.columns = [str(i)]

    # Append the yearly DataFrame to the annual_output_uncertainty_change DataFrame
    annual_output_uncertainty_const = pd.concat([annual_output_uncertainty_const, yearly], axis=1)
    #annual_output_uncertainty_const = annual_output_uncertainty_const.iloc[:, :-1]


# In[12]:


# NUMBER 10
#assuming low sellback price

import pandas as pd
import numpy as np

hour_effective_output = pd.DataFrame()

for i in range(df_efficiency.shape[1]):
    hour_effective_output[i] = averaged_irradiance.iloc[:, 0] * df_efficiency.iloc[:, i]

# convert watt-hours/hour to kilowatt-hours/day
day_effective_output = 24*hour_effective_output/1000 

# now we need to customize the panel specifications and the other parameters

price_rate = 0.041 # GOV SMART EXPORT GUARANTEE export rate 18p/KWH RANGE IS AROUND 3-24 
number_of_panels = 20    # years
lifetime = 25
size = 1.81    # meters squared THIS CAN BE INACCURATE CUZ NOT ALL SURFACE AREA IS THE PV CELL 
revenue = []
for column in day_effective_output:
    revenue.append(price_rate*day_effective_output[column].sum())
    
# starting from the 21st year 2021
real_revenue = [x*size for x in revenue]


print(f"Revenue Over 25 Years per 1 panel: {sum(real_revenue)} pounds")

lifetime_revenue = round(number_of_panels*sum(real_revenue),2)
print(f"Total Revenue Over 25 Years If we assume {number_of_panels} panels {lifetime_revenue} pounds")

annual_revenue = round(lifetime_revenue/lifetime,2)
print(f"Revenue Average per 1 year for {number_of_panels} panels: {annual_revenue} pounds")


# In[13]:


# NUMBER 11
#assuming high sellback price 

import pandas as pd
import numpy as np

hour_effective_output = pd.DataFrame()

for i in range(df_efficiency.shape[1]):
    hour_effective_output[i] = averaged_irradiance.iloc[:, 0] * df_efficiency.iloc[:, i]

# convert watt-hours/hour to kilowatt-hours/day
day_effective_output = 24*hour_effective_output/1000 

# now we need to customize the panel specifications and the other parameters

price_rate = 0.15 # GOV SMART EXPORT GUARANTEE export rate 18p/KWH RANGE IS AROUND 3-24 
number_of_panels = 20    
lifetime = 25
size = 1.81     # meters squared THIS CAN BE INACCURATE CUZ NOT ALL SURFACE AREA IS THE PV CELL 
revenue = []
for column in day_effective_output:
    revenue.append(price_rate*day_effective_output[column].sum())
    
# starting from the 21st year 2021
real_revenue = [x*size for x in revenue]

print(f"Revenue Over 25 Years per 1 panel: {sum(real_revenue)} pounds")

lifetime_revenue = round(number_of_panels*sum(real_revenue),2)
print(f"Total Revenue Over 25 Years If we assume {number_of_panels} panels {lifetime_revenue} pounds")

annual_revenue = round(lifetime_revenue/lifetime,2)
print(f"Revenue Average per 1 year for {number_of_panels} panels: {annual_revenue} pounds")


changed_temps_rev = real_revenue

day_effective_output_change = day_effective_output


# In[14]:


# Number 12
# Efficiency for constant temperature (frozen at year 2020)

import pandas as pd
import numpy as np
import matplotlib.cm as cm


df_efficiency_const = pd.DataFrame()
months = ['Jan', 'Apr', 'Jul', 'Oct']
month_positions = np.linspace(0, 365, num=4)

# Iterate over the columns of the temperature df
for iteration, year in enumerate(temps_projection_all.columns):
    # Get the temperature data for the year
    year_temps = temps_projection_all[year]
 
    
    # Calculate the efficiency for each day and store it
    df_efficiency_const[year] = np.round(
        initialEff + TC * (np.array(temps_projection_all.iloc[:,0]) - standardTestTemp) - (iteration) * (degradeRate), 2) / 100

fig = plt.figure(figsize=(10, 9))
cmap = cm.get_cmap('viridis')  # Change 'viridis' to any colormap you like

# iterate over the years to get efficiency 
for iteration, year in enumerate(df_efficiency_const.columns):
    # Only plot for every 5th year
    if int(year) % 5 == 0:
        # Get the color for the line based on its position
        color = cmap(float(iteration) / len(df_efficiency_const.columns))
        plt.plot(df_efficiency_const[year], color=color, label=year, linewidth= 3)

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=12)

plt.xticks(month_positions, months, fontsize=16)
plt.ylabel('Solar Panel Efficiency (%)', fontsize=16)
plt.title('Efficiency Throughout the Years (No Warming Factor)', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.savefig("C:/Users/tartn/Pictures/solarize/efficinecy_temps_const.png", dpi=400, bbox_inches='tight')
plt.show()


efficiency_uncertainity_const = df_efficiency_const*uncertainties_all
#annual_efficiency_uncertainties_const = efficiency_uncertainity_const.apply(lambda col: np.sqrt(np.sum(np.square(col))), axis=0)




import pandas as pd
import numpy as np

hour_effective_output = pd.DataFrame()


for i in range(df_efficiency.shape[1]):
    hour_effective_output[i] = averaged_irradiance.iloc[:, 0] * df_efficiency_const.iloc[:, i]
    


# convert watt-hours/hour to kilowatt-hours/day
# this is saying that it is producing energy X per hour 
day_effective_output = 24*hour_effective_output/1000 

day_effective_output_const = day_effective_output
# now we need to customize the panel specifications and the other parameters

price_rate = 0.15 # GOV SMART EXPORT GUARANTEE export rate 18p/KWH RANGE IS AROUND 3-24 
number_of_panels = 20    # years
lifetime = 25
size = 1.81     # meters squared THIS CAN BE INACCURATE CUZ NOT ALL SURFACE AREA IS THE PV CELL 
revenue = []
for column in day_effective_output:
    revenue.append(price_rate*day_effective_output[column].sum())
    
# starting from the 21st year 2021
const_real_revenue = [x*size for x in revenue]

print(f"Revenue Over 25 Years per 1 panel: {sum(const_real_revenue)} pounds")

const_lifetime_revenue = round(number_of_panels*sum(const_real_revenue),2)
print(f"Total Revenue Over 25 Years If we assume {number_of_panels} panels {const_lifetime_revenue} pounds")

const_annual_revenue = round(lifetime_revenue/lifetime,2)
print(f"Revenue Average per 1 year for {number_of_panels} panels: {const_annual_revenue} pounds")


# In[15]:


# Number 13

# THESE ARE THE SUM OF OUTPUT UNCERTAINITY EACH YEAR 
# factor in uncertainty the conversion to KWHr, size, price 


sum_annual_output_uncertainties_change= annual_output_uncertainty_change.apply(lambda col: np.sqrt(np.sum(np.square(col))), axis=0)
sum_annual_output_uncertainties_const = annual_output_uncertainty_const.apply(lambda col: np.sqrt(np.sum(np.square(col))), axis=0)

price_rate = 0.15
size = 1.81
# convert watt-hours/hour to kilowatt-hours/day
# since the irradiance was averaged with and divided by 24, we muliply it back to get 1 day average 
output_uncertainty_change = (24*sum_annual_output_uncertainties_change/1000)*price_rate*size
output_uncertainty_const = (24*sum_annual_output_uncertainties_const/1000)*price_rate*size

# for each panel 
best_output_change = real_revenue + output_uncertainty_change
worst_output_change = real_revenue - output_uncertainty_change
total_best_output_change = best_output_change.sum()
total_worst_output_change = worst_output_change.sum()


best_output_const = const_real_revenue + output_uncertainty_const
worst_output_const = const_real_revenue - output_uncertainty_const
total_best_output_const = sum(best_output_const)
total_worst_output_const = sum(worst_output_const)


# In[16]:


# Number 12
# Efficiency for constant temperature (frozen at year 2020)

import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

df_efficiency_const = pd.DataFrame()


# Iterate over the columns of the temperature df
for iteration, year in enumerate(temps_projection_all.columns):
    # Get the temperature data for the year
    year_temps = temps_projection_all[year]
 
    # Calculate the efficiency for each day and store it
    df_efficiency_const[year] = np.round(
        initialEff + TC * (np.array(temps_projection_all.iloc[:,0]) - standardTestTemp) - (iteration) * (degradeRate), 2) / 100

# NUMBER 8

# Final Efficiency over the years (if temperature increases)

df_efficiency = pd.DataFrame()

# Iterate over the columns in the original dataframe
for iteration, year in enumerate(temps_projection_all.columns):
    # Get the temperature data for the year
    year_temps = temps_projection_all[year]
 
    # Calculate the efficiency for each day and store it in the new dataframe
    df_efficiency[year] = np.round(initialEff + TC * (year_temps - standardTestTemp) - (iteration) * (degradeRate), 2) / 100

# Create a figure with 2 subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
cmap = cm.get_cmap('viridis')  # Change 'viridis' to any colormap you like

# Plot the first subplot
for iteration, year in enumerate(df_efficiency_const.columns):
    if int(year) % 5 == 0:
        color = cmap(float(iteration) / len(df_efficiency_const.columns))
        axs[0].plot(df_efficiency_const[year], color=color, linewidth=3)
        
months = ['Jan', 'Apr', 'Jul', 'Oct', 'Dec']
month_positions = np.linspace(0, 330, num=len(months))

axs[0].set_xticks(month_positions)
yticks = axs[0].get_yticks()
axs[0].set_yticks(yticks)
axs[1].set_yticks(yticks)

axs[0].set_xticklabels(months, fontsize=18)  # Increase font size
axs[0].set_xlabel('Months', fontsize=18)  # Increase font size
axs[0].set_ylabel('Solar Panel Efficiency (%)', fontsize=18)  # Increase font size
axs[0].set_title('Efficiency (No Warming Factor)', fontsize=20)  # Increase font size
axs[0].tick_params(axis='both', which='major', labelsize=16)  # Increase font size
for y in axs[0].get_yticks():
    axs[0].axhline(y, color='gray', linestyle='dashed', alpha=0.7)
axs[0].set_xlim(left=0)
axs[0].set_xlim(left=0, right=df_efficiency.index[-1])

# Plot the second subplot
for iteration, year in enumerate(df_efficiency.columns):
    if int(year) % 5 == 0:
        color = cmap(float(iteration) / len(df_efficiency.columns))
        axs[1].plot(df_efficiency[year], color=color, linewidth=3, label=year)

axs[1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=16)  # Increase font size
axs[1].set_xticks(month_positions)
axs[1].set_xticklabels(months, fontsize=18)  # Increase font size
axs[1].set_xlabel('Months', fontsize=18)  # Increase font size
for y in axs[1].get_yticks():
    axs[1].axhline(y, color='gray', linestyle='dashed', alpha=0.7)
axs[1].set_title('Efficiency (Warming Factored)', fontsize=20)  # Increase font size
axs[1].tick_params(axis='both', which='major', labelsize=16)  # Increase font size
axs[1].set_xlim(left=0)
axs[1].set_xlim(left=0, right=df_efficiency.index[-1])

plt.tight_layout() # this is to prevent the overlapping 
plt.savefig("C:/Users/tartn/Pictures/solarize/efficinecy_comparison_of_both.png", dpi=400, bbox_inches='tight')

plt.show()


# In[17]:


# Number 14
# plot the results 

import matplotlib.pyplot as plt

x = range(len(const_real_revenue[:-1]))

fig = plt.figure(figsize=(10, 6))
plt.xlim(x[0], x[-1])
plt.plot(x, changed_temps_rev[:-1], label='Global Warming Trend',color='orange')
plt.fill_between(x, real_revenue[:-1] - output_uncertainty_change[:-1], real_revenue[:-1] + output_uncertainty_change[:-1], color='red', alpha=0.1)

plt.plot(x, const_real_revenue[:-1], label='Constant Temperature Trend',color='blue')
plt.fill_between(x, const_real_revenue[:-1] - output_uncertainty_const[:-1], const_real_revenue[:-1] + output_uncertainty_const[:-1], color='blue', alpha=0.2)


plt.xlabel('Years', fontsize=16)  
years = np.arange(2020, 2046, 5)
plt.xticks(np.arange(0, len(years)*5, 5), years)
ax.set_xlabel('Years')

plt.ylabel('Annual revenue (pounds)', fontsize=16)  
plt.title('Revenue Comparison Per Panel at Kew Gardens', fontsize=18)  
plt.legend(fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid()
plt.savefig("C:/Users/tartn/Pictures/solarize/results.png", dpi=400)
plt.show()

print('\nConstant Temperature')
print(f"Total Revenue Over 25 Years If we assume {number_of_panels} panels {const_lifetime_revenue} pounds'")
print(f"Best Case Rev: {round(number_of_panels*total_best_output_const,2)} pounds")
print(f"Worst Case Rev: {round(number_of_panels*total_worst_output_const,2)} pounds")

print('\nGlobal Warming')
print(f"Total Revenue Over 25 Years If we assume {number_of_panels} panels {lifetime_revenue} pounds")
print(f"Best Case Rev: {round(number_of_panels*total_best_output_change,2)} pounds")
print(f"Worst Case Rev: {round(number_of_panels*total_worst_output_change,2)} pounds")


print(f'\nLifetime Total Revenue Difference: {np.round(const_lifetime_revenue-lifetime_revenue,2)} pounds ')

print('\nThe constant graph downtrend is from the degredation rate + effect of varying temp of 2020 ')
print('The changed graph downtrend is from the degredation rate + effect of varying temp of 2020 + global warming\n')
print('constant temp has lower uncertainty compared to warming trend')

